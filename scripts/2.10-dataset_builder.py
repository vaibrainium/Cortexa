"""
dataset_builder.py — Automated instruction pair generation for Cortexa.

Pipeline per pair:
    1. Take a seed question (synthesis / gap-finding / hypothesis)
    2. Retrieve top-5 chunks from Qdrant via BGE-M3 dense search
    3. Send chunks + question to Claude API to generate a structured answer
    4. Save as pairs.jsonl in HF Dataset format

Target: 120 pairs
    - 50 synthesis   (cross-subfield: "how does X relate to Y?")
    - 40 gap-finding (known gaps from corpus map)
    - 30 hypothesis  (anchored to 4 thesis chapters)

Output schema (one JSON line per pair):
    instruction   str   the question / prompt
    input         str   "" (empty -- context is baked into the answer via RAG)
    output        str   the model answer with inline author/year citations
    pair_type     str   "synthesis" | "gap" | "hypothesis"
    subfields     list  subfields involved
    chunk_ids     list  chunk_ids of retrieved context used to generate answer

Usage:
    python src/dataset_builder.py
    python src/dataset_builder.py --dry-run      # generate 3 pairs, don't write
    python src/dataset_builder.py --limit 10     # generate first N pairs only
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import anthropic
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from config import dir_config
ROOT_DIR = Path(dir_config.data.root)
PROCESSED_DIR = ROOT_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

OUTPUT_PATH = PROCESSED_DIR / "pairs.jsonl"

QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "cortexa"
BGE_MODEL_NAME = "BAAI/bge-m3"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K = 5          # chunks retrieved per question
RATE_LIMIT_DELAY = 1.0   # seconds between API calls to avoid rate limiting

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed questions
# 120 total: 50 synthesis, 40 gap, 30 hypothesis
# Subfield keys match corpus map:
#   value_based_dm, perceptual_dm, rl_brain, drift_diffusion,
#   neuroeconomics, cognitive_control, confidence_metacognition, sc_subcortical
# ---------------------------------------------------------------------------

SYNTHESIS_QUESTIONS = [
    # SC x Perceptual DM
    {"q": "How does Superior Colliculus activity relate to evidence accumulation during perceptual decision-making?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "What is the computational role of the Superior Colliculus in the drift-diffusion model framework?", "subfields": ["sc_subcortical", "drift_diffusion"]},
    {"q": "How do buildup neurons in the Superior Colliculus compare to ramping activity in LIP during decision formation?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "How does SC inactivation affect perceptual decision thresholds and what does this tell us about its role in evidence accumulation?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "What is the relationship between target probability encoding in the Superior Colliculus and Bayesian prior implementation?", "subfields": ["sc_subcortical", "perceptual_dm"]},

    # SC x RL
    {"q": "How does the Superior Colliculus encode learned probability across trials and how does this relate to reinforcement learning?", "subfields": ["sc_subcortical", "rl_brain"]},
    {"q": "What is the relationship between dopaminergic modulation of the basal ganglia and Superior Colliculus activity during decision-making?", "subfields": ["sc_subcortical", "rl_brain"]},
    {"q": "How do the basal ganglia and Superior Colliculus interact during action selection and value-based decisions?", "subfields": ["sc_subcortical", "rl_brain"]},

    # SC x Value-based DM
    {"q": "How does the Superior Colliculus contribute to value-based decision-making beyond its classical role as a motor relay?", "subfields": ["sc_subcortical", "value_based_dm"]},
    {"q": "How do OFC value signals interact with Superior Colliculus activity during target selection?", "subfields": ["sc_subcortical", "value_based_dm"]},

    # SC x Cognitive Control
    {"q": "How does top-down control from prefrontal cortex modulate Superior Colliculus activity during flexible decision-making?", "subfields": ["sc_subcortical", "cognitive_control"]},
    {"q": "What is the role of the Superior Colliculus in attentional control and how does this interact with executive function?", "subfields": ["sc_subcortical", "cognitive_control"]},

    # SC x Confidence/Metacognition
    {"q": "Does the Superior Colliculus encode decision confidence, and how does this relate to metacognitive monitoring?", "subfields": ["sc_subcortical", "confidence_metacognition"]},
    {"q": "How do subcortical structures like the Superior Colliculus contribute to the neural basis of certainty signals?", "subfields": ["sc_subcortical", "confidence_metacognition"]},

    # SC x Neuroeconomics
    {"q": "How does the Superior Colliculus implement economic choice computations beyond simple sensorimotor transformation?", "subfields": ["sc_subcortical", "neuroeconomics"]},
    {"q": "What does the Duan et al. 2024 finding about abstract category encoding in SC mean for neuroeconomic theories of decision-making?", "subfields": ["sc_subcortical", "neuroeconomics"]},

    # Perceptual DM x RL
    {"q": "How do reinforcement learning and perceptual decision-making share computational mechanisms in the brain?", "subfields": ["perceptual_dm", "rl_brain"]},
    {"q": "What is the relationship between reward prediction errors and perceptual decision thresholds?", "subfields": ["perceptual_dm", "rl_brain"]},
    {"q": "How does prior probability learning in perceptual tasks relate to model-based reinforcement learning?", "subfields": ["perceptual_dm", "rl_brain"]},
    {"q": "How does the drift-diffusion model relate to temporal difference learning algorithms in reinforcement learning?", "subfields": ["drift_diffusion", "rl_brain"]},

    # Perceptual DM x DDM
    {"q": "How does the drift-diffusion model account for the neural mechanisms of perceptual choice in LIP and MT?", "subfields": ["perceptual_dm", "drift_diffusion"]},
    {"q": "How do starting point biases and drift-rate offsets in the DDM map onto different neural mechanisms of prior implementation?", "subfields": ["perceptual_dm", "drift_diffusion"]},
    {"q": "What is the relationship between psychometric function parameters and drift-diffusion model parameters in perceptual tasks?", "subfields": ["perceptual_dm", "drift_diffusion"]},

    # Perceptual DM x Value-based DM
    {"q": "How do perceptual decision-making and value-based decision-making share and differ in their neural substrates?", "subfields": ["perceptual_dm", "value_based_dm"]},
    {"q": "What is the relationship between sensory evidence accumulation and subjective value computation during decisions under uncertainty?", "subfields": ["perceptual_dm", "value_based_dm"]},

    # Perceptual DM x Cognitive Control
    {"q": "How does cognitive control modulate decision thresholds in perceptual decision-making tasks?", "subfields": ["perceptual_dm", "cognitive_control"]},
    {"q": "What is the role of ACC in adjusting decision criteria during perceptual tasks with changing statistics?", "subfields": ["perceptual_dm", "cognitive_control"]},

    # Perceptual DM x Confidence
    {"q": "How does decision confidence relate to the quality of sensory evidence accumulation in perceptual tasks?", "subfields": ["perceptual_dm", "confidence_metacognition"]},
    {"q": "What neural mechanisms link perceptual choice accuracy to post-decisional confidence signals?", "subfields": ["perceptual_dm", "confidence_metacognition"]},

    # RL x DDM
    {"q": "How can drift-diffusion model parameters be used to characterize reinforcement learning strategies across trials?", "subfields": ["rl_brain", "drift_diffusion"]},
    {"q": "How does the STN contribute to decision thresholds in both the DDM and basal ganglia reinforcement learning circuits?", "subfields": ["rl_brain", "drift_diffusion"]},

    # RL x Cognitive Control
    {"q": "How do model-based and model-free reinforcement learning systems interact with prefrontal cognitive control?", "subfields": ["rl_brain", "cognitive_control"]},
    {"q": "What is the expected value of control framework and how does it relate to reinforcement learning in the brain?", "subfields": ["rl_brain", "cognitive_control"]},

    # RL x Neuroeconomics
    {"q": "How does prospect theory relate to neural reward prediction error signals in the striatum?", "subfields": ["rl_brain", "neuroeconomics"]},
    {"q": "How do temporal difference learning and neuroeconomic utility functions relate to dopamine signaling?", "subfields": ["rl_brain", "neuroeconomics"]},

    # RL x Confidence
    {"q": "How does uncertainty estimation in reinforcement learning relate to metacognitive confidence signals?", "subfields": ["rl_brain", "confidence_metacognition"]},

    # DDM x Cognitive Control
    {"q": "How does cognitive control modulate the speed-accuracy tradeoff captured by the drift-diffusion model?", "subfields": ["drift_diffusion", "cognitive_control"]},
    {"q": "How does the leaky competing accumulator model relate to neural mechanisms of cognitive control?", "subfields": ["drift_diffusion", "cognitive_control"]},

    # DDM x Neuroeconomics
    {"q": "How can drift-diffusion model parameters be interpreted in terms of neuroeconomic utility and value normalization?", "subfields": ["drift_diffusion", "neuroeconomics"]},

    # Value x Cognitive Control
    {"q": "How does prefrontal cortex arbitrate between competing value signals during economic choice?", "subfields": ["value_based_dm", "cognitive_control"]},
    {"q": "How does motivation interact with cognitive control to regulate value-based decision-making?", "subfields": ["value_based_dm", "cognitive_control"]},

    # Value x Confidence
    {"q": "How does decision confidence modulate value-based choice and willingness to pay?", "subfields": ["value_based_dm", "confidence_metacognition"]},

    # Neuroeconomics x Cognitive Control
    {"q": "How do executive functions constrain economic rationality in human decision-making?", "subfields": ["neuroeconomics", "cognitive_control"]},

    # Neuroeconomics x Confidence
    {"q": "How does metacognitive uncertainty influence risk preferences and economic choice?", "subfields": ["neuroeconomics", "confidence_metacognition"]},

    # Confidence x Cognitive Control
    {"q": "How does metacognitive monitoring interact with cognitive control to adjust decision strategies?", "subfields": ["confidence_metacognition", "cognitive_control"]},

    # Three-way
    {"q": "How do the SC, basal ganglia, and prefrontal cortex interact to implement flexible decision-making under changing reward contingencies?", "subfields": ["sc_subcortical", "rl_brain", "cognitive_control"]},
    {"q": "How do evidence accumulation, value computation, and confidence interact during decisions under uncertainty?", "subfields": ["perceptual_dm", "value_based_dm", "confidence_metacognition"]},
    {"q": "How does the DDM framework unify perceptual decision-making, reinforcement learning, and cognitive control?", "subfields": ["drift_diffusion", "perceptual_dm", "rl_brain"]},
    {"q": "How does dopamine modulate both reinforcement learning and perceptual prior implementation, and what does this mean for Parkinson's disease?", "subfields": ["rl_brain", "perceptual_dm", "sc_subcortical"]},
    {"q": "How does the Superior Colliculus integrate sensory evidence, prior probability, and action value to produce decisions?", "subfields": ["sc_subcortical", "perceptual_dm", "value_based_dm"]},
]

GAP_QUESTIONS = [
    # Gap 1: RL and perceptual DM don't cite each other
    {"q": "What is the gap between reinforcement learning and perceptual decision-making research, and why do these fields rarely integrate despite shared computational mechanisms?", "subfields": ["rl_brain", "perceptual_dm"]},
    {"q": "How could drift-diffusion model formalisms be applied to characterize learning dynamics in reinforcement learning tasks?", "subfields": ["drift_diffusion", "rl_brain"]},
    {"q": "What would a unified computational account of perceptual learning and reward learning look like?", "subfields": ["perceptual_dm", "rl_brain"]},
    {"q": "Why has the accumulation-to-threshold framework from perceptual DM not been widely adopted in the reinforcement learning literature?", "subfields": ["perceptual_dm", "rl_brain", "drift_diffusion"]},
    {"q": "How do trial-by-trial learning signals from reinforcement learning theories map onto sequential sampling mechanisms in perceptual decisions?", "subfields": ["rl_brain", "perceptual_dm", "drift_diffusion"]},

    # Gap 2: SC largely absent from computational DM reviews
    {"q": "Why is the Superior Colliculus largely absent from mainstream computational accounts of decision-making, and what evidence suggests it should be included?", "subfields": ["sc_subcortical", "drift_diffusion", "perceptual_dm"]},
    {"q": "How should the Jun et al. 2021 finding that SC causally contributes to evidence computation be integrated into drift-diffusion model accounts?", "subfields": ["sc_subcortical", "drift_diffusion"]},
    {"q": "What computational role should subcortical structures play in models of perceptual decision-making that currently focus only on cortical areas?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "How does the SC-LIP-PFC pathway challenge purely cortical accounts of evidence accumulation?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "What is missing in our understanding of how subcortical decision circuits interact with cortical accumulation mechanisms?", "subfields": ["sc_subcortical", "perceptual_dm", "cognitive_control"]},

    # Gap 3: Dopamine links RL and perceptual DM but two literatures don't engage
    {"q": "How does dopamine modulate both reward prediction errors in RL and perceptual prior implementation, and why do these two literatures rarely connect?", "subfields": ["rl_brain", "perceptual_dm"]},
    {"q": "What does Parkinson's disease tell us about the shared dopaminergic mechanisms underlying reinforcement learning and perceptual decision-making?", "subfields": ["rl_brain", "perceptual_dm"]},
    {"q": "How could studying dopamine depletion in Parkinson's disease bridge the gap between reinforcement learning and perceptual prior implementation research?", "subfields": ["rl_brain", "perceptual_dm"]},
    {"q": "What is the relationship between tonic and phasic dopamine signals in reinforcement learning and their role in setting perceptual decision thresholds?", "subfields": ["rl_brain", "perceptual_dm", "drift_diffusion"]},

    # Gap 4: DDM underused in RL and value literatures
    {"q": "How could the drift-diffusion model framework be applied to value-based decision-making to better characterize the dynamics of economic choice?", "subfields": ["drift_diffusion", "value_based_dm"]},
    {"q": "What is the gap between formal computational models of perceptual decisions and the less formal accounts used in neuroeconomics?", "subfields": ["drift_diffusion", "neuroeconomics"]},
    {"q": "How could hierarchical DDM approaches like HDDM be applied to reinforcement learning datasets to extract learning-related changes in accumulation dynamics?", "subfields": ["drift_diffusion", "rl_brain"]},
    {"q": "Why has the neuroeconomics field not widely adopted sequential sampling frameworks despite their success in perceptual decision-making?", "subfields": ["neuroeconomics", "drift_diffusion", "perceptual_dm"]},

    # Gap 5: Comparative mouse vs primate decision strategies
    {"q": "What is the gap in our understanding of whether mice and primates use the same evidence accumulation mechanisms during perceptual decisions?", "subfields": ["perceptual_dm", "sc_subcortical"]},
    {"q": "How could comparative studies across mice and non-human primates using the same task resolve questions about the evolution of decision circuits?", "subfields": ["perceptual_dm", "sc_subcortical"]},
    {"q": "Do mice accumulate evidence over time like primates, or do they rely on instantaneous evidence samples, and what does this mean for the generalizability of primate decision models?", "subfields": ["perceptual_dm", "sc_subcortical"]},
    {"q": "What is the causal role of Wide Field Vertical cells in the mouse Superior Colliculus during perceptual decision-making, and how does this compare to primate SC function?", "subfields": ["sc_subcortical", "perceptual_dm"]},

    # Gap 6: SC/subcortical literature absent from cognitive control and metacognition
    {"q": "How might subcortical structures like the Superior Colliculus contribute to cognitive control functions that are typically attributed exclusively to prefrontal cortex?", "subfields": ["sc_subcortical", "cognitive_control"]},
    {"q": "What is the gap in our understanding of how subcortical decision circuits contribute to metacognitive monitoring and confidence?", "subfields": ["sc_subcortical", "confidence_metacognition"]},
    {"q": "How does the finding that SC encodes abstract categories (Duan et al. 2024) challenge the view that higher cognitive functions require prefrontal cortex?", "subfields": ["sc_subcortical", "cognitive_control"]},

    # General cross-field gaps
    {"q": "What computational principles are shared between value normalization in neuroeconomics and gain control in perceptual decision-making?", "subfields": ["neuroeconomics", "perceptual_dm"]},
    {"q": "How does the confidence literature fail to account for subcortical contributions to certainty signals?", "subfields": ["confidence_metacognition", "sc_subcortical"]},
    {"q": "What is missing in current accounts of cognitive control that would be revealed by studying subcortical contributions to executive function?", "subfields": ["cognitive_control", "sc_subcortical", "rl_brain"]},
    {"q": "How do current neuroeconomic models fail to account for the role of evidence accumulation dynamics in shaping economic preferences?", "subfields": ["neuroeconomics", "drift_diffusion", "perceptual_dm"]},
    {"q": "What would a complete mechanistic account of Bayesian prior implementation look like, spanning subcortical encoding, accumulation dynamics, and behavioral expression?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "How does the field of cognitive control fail to engage with subcortical and reinforcement learning accounts of flexible behavior?", "subfields": ["cognitive_control", "rl_brain", "sc_subcortical"]},
    {"q": "What are the key open questions at the intersection of confidence, reinforcement learning, and perceptual decision-making?", "subfields": ["confidence_metacognition", "rl_brain", "perceptual_dm"]},
    {"q": "How could the neuroeconomics framework be extended to account for the role of subcortical structures in economic valuation?", "subfields": ["neuroeconomics", "sc_subcortical", "value_based_dm"]},
    {"q": "What is the gap between single-neuron electrophysiology accounts of decision-making and population-level computational models?", "subfields": ["perceptual_dm", "drift_diffusion", "sc_subcortical"]},
    {"q": "How does the lack of cross-species comparative work limit our understanding of the evolution of decision-making circuits?", "subfields": ["perceptual_dm", "sc_subcortical", "rl_brain"]},
    {"q": "What would a unified account of implicit and explicit prior learning look like at the neural circuit level?", "subfields": ["perceptual_dm", "rl_brain", "sc_subcortical"]},
    {"q": "How do current models of cognitive control fail to account for the contribution of basal ganglia and SC to flexible decision-making?", "subfields": ["cognitive_control", "sc_subcortical", "rl_brain"]},
]

HYPOTHESIS_QUESTIONS = [
    # Chapter 1 -- Implicit/explicit prior learning in humans
    {"q": "Based on the finding that implicit prior learners show drift-rate offsets rather than starting point biases, what does this predict about the neural locus of prior implementation in SC versus cortex?", "subfields": ["perceptual_dm", "drift_diffusion", "sc_subcortical"]},
    {"q": "Given that humans can learn priors implicitly without awareness, what does this predict about the subcortical versus cortical locus of prior storage?", "subfields": ["perceptual_dm", "sc_subcortical", "rl_brain"]},
    {"q": "If implicit prior learning involves drift-rate offsets in the DDM, what does this predict about the relative contributions of SC buildup neurons versus LIP ramping activity?", "subfields": ["perceptual_dm", "drift_diffusion", "sc_subcortical"]},
    {"q": "What does the existence of multiple learner types (implicit, partial explicit, explicit) predict about the multiplicity of neural mechanisms for prior implementation?", "subfields": ["perceptual_dm", "rl_brain", "cognitive_control"]},
    {"q": "If explicit prior awareness involves different DDM parameters than implicit learning, what does this predict about the role of prefrontal cortex in prior implementation?", "subfields": ["perceptual_dm", "drift_diffusion", "cognitive_control"]},
    {"q": "Given that stimulus-specific priors are implemented via drift-rate rather than starting point, what does this predict about the timing of prior effects on SC neural activity?", "subfields": ["perceptual_dm", "sc_subcortical", "drift_diffusion"]},
    {"q": "What testable predictions does the implicit prior learning framework make about confidence calibration across different learner types?", "subfields": ["perceptual_dm", "confidence_metacognition"]},

    # Chapter 2 -- Parkinson's disease and dopamine
    {"q": "If dopamine medication restores prior learning in Parkinson's disease, what does this predict about the specific DDM parameters affected by dopaminergic depletion?", "subfields": ["rl_brain", "perceptual_dm", "drift_diffusion"]},
    {"q": "Given that Tremor Dominant and Bradykinetic Dominant PD subtypes may show different responses to dopamine medication, what does this predict about the distinct dopaminergic pathways involved in prior implementation?", "subfields": ["rl_brain", "perceptual_dm"]},
    {"q": "If dopamine depletion impairs prior learning via striatal mechanisms, what does this predict about the SC's role in receiving and implementing prior signals from basal ganglia?", "subfields": ["rl_brain", "sc_subcortical", "perceptual_dm"]},
    {"q": "What does the differential effect of dopamine medication on PD subtypes predict about the role of direct versus indirect basal ganglia pathways in prior implementation?", "subfields": ["rl_brain", "perceptual_dm", "drift_diffusion"]},
    {"q": "If PD patients show impaired prior learning that correlates with disease severity, what does this predict about the relationship between dopamine tone and decision threshold setting in the DDM?", "subfields": ["rl_brain", "drift_diffusion", "perceptual_dm"]},
    {"q": "Based on the dopaminergic modulation of prior learning in PD, what does this predict about the shared versus distinct mechanisms of model-based and model-free RL in prior implementation?", "subfields": ["rl_brain", "perceptual_dm", "cognitive_control"]},

    # Chapter 3 -- SC encoding of priors in NHP
    {"q": "Given that dPCA reveals a latent state (bias) PC in SC population activity, what does this predict about the specific cell types in SC that encode prior probability?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "If SC ensembles transition from bias encoding to choice encoding over trial time, what does this predict about the causal role of SC in implementing versus reading out priors?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "Based on the GLM-HMM finding that monkeys alternate between biased and unbiased latent states, what does this predict about the stability of prior representations in SC across state transitions?", "subfields": ["sc_subcortical", "perceptual_dm", "rl_brain"]},
    {"q": "If the distance between biased and unbiased population subspaces in SC correlates with behavioral prior strength, what does this predict about the geometry of prior representations across brain areas?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "Given that SC activity is heterogeneous at the single-neuron level but structured at the population level, what does this predict about the optimal readout mechanism for prior signals downstream?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "If top-down prior signals from PFC modulate SC activity, what specific neural signatures in SC would distinguish a top-down prior from a sensory-driven bias?", "subfields": ["sc_subcortical", "cognitive_control", "perceptual_dm"]},
    {"q": "What does the subspace analysis of SC population activity predict about how prior and choice information are multiplexed or separated in downstream motor areas?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "Based on Basso & Wurtz 1998 showing buildup neurons encode target probability, what does this predict about how the SC updates its prior representation within versus across blocks?", "subfields": ["sc_subcortical", "perceptual_dm", "rl_brain"]},

    # Chapter 4 -- Mouse comparative / WFV cells
    {"q": "If mice use instantaneous evidence rather than accumulation during random dot motion tasks, what does this predict about the role of SC versus cortex in their decision computation?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "Given that unilateral SC inactivation reduces contralateral choices in mice, what does this predict about the specific contribution of WFV cells to the decision bias?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "If WFV cells are causally involved in perceptual decisions in mice, what does this predict about their homologs in primate SC and their role in prior encoding?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "Based on the comparative framework between mice and NHP, what does the presence or absence of evidence accumulation across species predict about the minimum circuit requirements for Bayesian prior implementation?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "If mice show different decision strategies than primates on the same task, what does this predict about the evolutionary conservation of SC-based decision circuits?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "Given the fast subcortical MT-SC-motor route available in mice, what does this predict about when mice would show evidence accumulation versus instantaneous evidence use?", "subfields": ["sc_subcortical", "perceptual_dm", "drift_diffusion"]},
    {"q": "What does the causal role of WFV cells in mouse SC predict about the circuit mechanism by which prior probability biases perceptual choices?", "subfields": ["sc_subcortical", "perceptual_dm", "rl_brain"]},
    {"q": "If SC inactivation produces a choice bias equivalent to a starting point shift in the DDM, what does this predict about the timing and nature of SC's contribution to decision formation?", "subfields": ["sc_subcortical", "drift_diffusion", "perceptual_dm"]},
    {"q": "Based on the within-subject control design for WFV inactivation, what would a null result versus a specific deficit predict about the necessity versus sufficiency of SC for prior implementation?", "subfields": ["sc_subcortical", "perceptual_dm"]},
    {"q": "Given that both primate SC (Ch.3) and mouse SC (Ch.4) appear to encode priors, what does this cross-species conservation predict about the ancestral function of the SC in decision-making?", "subfields": ["sc_subcortical", "perceptual_dm", "rl_brain"]},
]

# ---------------------------------------------------------------------------
# System prompt for Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Cortexa, a research assistant specialized in decision neuroscience.
You synthesize across subfields including perceptual decision-making, reinforcement learning,
drift-diffusion modeling, value-based decision-making, neuroeconomics, cognitive control,
confidence and metacognition, and the role of the Superior Colliculus in decision computation.

When answering questions:
- Always name which subfields and research communities are relevant
- Use each community's own terminology correctly
- Cite specific papers inline using (Author et al., Year) or (Author & Author, Year) format
- Surface agreements AND tensions between communities
- Connect brain regions to their computational roles explicitly
- For hypothesis questions: make predictions specific and falsifiable
- For gap questions: explain WHY the gap exists, not just that it exists
- Write 3-5 paragraphs. Be precise and technical -- the reader is a neuroscience PhD student.
- Do not hedge excessively. Make clear claims grounded in the retrieved literature.
- Do not summarize one paper. Synthesize across multiple papers and subfields."""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(query: str, model: SentenceTransformer, client: QdrantClient, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-k chunks from Qdrant for a given query."""
    vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "chunk_id": r.payload["chunk_id"],
            "text": r.payload["text"],
            "title": r.payload.get("title", ""),
            "first_author": r.payload.get("first_author", ""),
            "year": r.payload.get("year", ""),
            "subfield": r.payload.get("subfield", ""),
            "score": r.score,
        }
        for r in results
    ]


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the Claude prompt."""
    parts = []
    for i, c in enumerate(chunks):
        author = c["first_author"]
        year = c["year"]
        title = c["title"][:80]
        parts.append(
            f"[Source {i+1}] {author} et al. ({year}) -- {title}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    chunks: list[dict],
    anthropic_client: anthropic.Anthropic,
) -> str:
    """Call Claude API to generate a structured answer grounded in retrieved chunks."""
    context = format_context(chunks)
    user_message = f"""Retrieved literature context:

{context}

---

Question: {question}

Answer based on the retrieved context above. Use inline citations (Author et al., Year).
Synthesize across subfields -- do not summarize a single paper."""

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(limit: int | None = None, dry_run: bool = False):
    # --- Check API key ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    # --- Load models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading BGE-M3 on {device} ...")
    embed_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    if device == "cuda":
        embed_model = embed_model.half()
    log.info("BGE-M3 loaded")

    qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    # --- Build question list ---
    all_questions = (
        [{"type": "synthesis", **q} for q in SYNTHESIS_QUESTIONS] +
        [{"type": "gap", **q} for q in GAP_QUESTIONS] +
        [{"type": "hypothesis", **q} for q in HYPOTHESIS_QUESTIONS]
    )
    if limit:
        all_questions = all_questions[:limit]

    log.info(f"Generating {len(all_questions)} pairs "
             f"({len(SYNTHESIS_QUESTIONS)} synthesis, {len(GAP_QUESTIONS)} gap, "
             f"{len(HYPOTHESIS_QUESTIONS)} hypothesis)")

    # --- Output file ---
    writer = None
    if not dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        writer = open(OUTPUT_PATH, "w")

    pairs_written = 0
    errors = 0

    try:
        for item in tqdm(all_questions, desc="Generating pairs"):
            question = item["q"]
            pair_type = item["type"]
            subfields = item["subfields"]

            try:
                # Retrieve
                chunks = retrieve_chunks(question, embed_model, qdrant)

                if dry_run:
                    log.info(f"[DRY RUN] {pair_type}: {question[:80]}...")
                    log.info(f"  Retrieved: {[c['first_author'] + ' ' + c['year'] for c in chunks]}")
                    pairs_written += 1
                    continue

                # Generate
                answer = generate_answer(question, chunks, anthropic_client)

                # Build record
                record = {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "pair_type": pair_type,
                    "subfields": subfields,
                    "chunk_ids": [c["chunk_id"] for c in chunks],
                }
                writer.write(json.dumps(record) + "\n")
                pairs_written += 1

                # Rate limit
                time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                log.error(f"Failed on question: {question[:60]}... -- {e}")
                errors += 1
                continue

    finally:
        if writer:
            writer.close()

    log.info("=" * 50)
    log.info(f"Pairs generated  : {pairs_written}")
    log.info(f"Errors           : {errors}")
    if not dry_run:
        log.info(f"Output           : {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortexa dataset builder")
    parser.add_argument("--dry-run", action="store_true",
                        help="Retrieve chunks for each question but skip Claude API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate first N pairs (useful for testing)")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)

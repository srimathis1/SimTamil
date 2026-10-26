# SimTamil — Participant-Grounded AI Simulation

**Simulating individual attitudes and social behaviours in Tamil populations using grounded generative AI agents.**

SimTamil is a research project that explores how large language models (LLMs) can simulate individual-level human responses by grounding AI agents in participant interviews, survey responses, personality assessments, and contextual information.

The project covers the complete workflow, from participant data collection and preprocessing to AI-based simulation and evaluation. **The simulation component is currently under active development.**

## Live Applications

- **[SimTamil — Wave 2 Simulation](https://simtamil-wave2-2lqu.vercel.app/)** — Simulation interface.
- **[SimTamil — Data Collection](https://simule-data-collect.vercel.app/)** — Participant data collection interface.

## Project Workflow

```text
Participant Data Collection
          ↓
Audio Processing and Transcription
          ↓
Participant Profile Construction
          ↓
Expert Reflection Generation
          ↓
Domain Classification and Routing
          ↓
Participant-Grounded AI Simulation
          ↓
Evaluation Against Human Responses
```

## Key Components

### 1. Participant Data Collection
Collects participant information through Tamil voice interviews, localized social survey questions, personality assessments, and contextual information.

### 2. Audio Processing and Transcription
Processes interview recordings and converts Tamil speech into text for subsequent analysis and participant profiling.

### 3. Participant Profiling
Organizes available interview transcripts, demographic context, survey responses, and BFI-44 personality scores into participant-specific information.

### 4. Expert Reflection Generation
Uses multiple analytical perspectives to interpret participant information:

- **Demographer:** Population and socioeconomic context.
- **Psychologist:** Personality and individual attitudes.
- **Behavioural Economist:** Decision-making and economic choices.
- **Regional Political Scientist:** Public opinion and governance.
- **Sociologist / Cultural Anthropologist:** Social relationships and cultural context.

These are AI-generated expert perspectives, not independent human assessments.

### 5. Domain Classification and Routing
Identifies the expert perspective most relevant to an evaluation question and provides the corresponding context for response generation.

### 6. Participant-Grounded AI Simulation
Generates responses using the participant's available information and relevant expert reflections. This component is under active development, with ongoing work on simulation quality and evaluation.

### 7. Evaluation
Explores how closely AI-generated responses correspond to actual participant responses. Evaluation methods may include Pearson correlation for personality scores, Fisher's z-transformation for statistical analysis of correlations, and appropriate measures for other response types.

## Technology Stack

| Component | Technology |
|---|---|
| Language models | Gemini 2.5 Pro, Gemini 2.5 Flash |
| Audio processing | FFmpeg |
| Tamil transcription | IndicConformer |
| Structured outputs | Pydantic |
| Data storage | Supabase |
| Deployment | Vercel |

## Research Objectives

- Develop participant-grounded AI agents using individual-level information.
- Explore culturally relevant simulation of attitudes and social behaviours in Tamil populations.
- Investigate the role of personality, lived experiences, and social context in AI-generated responses.
- Evaluate simulated responses against actual human responses.
- Explore reliability, demographic representation, privacy, and population-level statistical fidelity.

## Current Development Status

**Status: Research prototype — simulation under active development.**

The project brings together participant data collection, data processing, participant profiling, expert reflection generation, and an AI-based simulation workflow. Further development focuses on improving simulation reliability, expanding participant representation, strengthening evaluation, and exploring privacy-preserving deployment.

The existence of a working interface does not, by itself, establish that the simulation has been validated. Performance claims should be based on completed experiments and reported evaluation results.

## Future Work

- Expand the participant cohort and demographic representation.
- Conduct longitudinal evaluations beyond the initial follow-up period.
- Benchmark open-weight language models against API-based models.
- Explore privacy-preserving, locally hosted inference.
- Evaluate population-level covariance and inter-item correlations.
- Investigate validated synthetic-cohort research designs.

## Research Ethics

Participant interviews and survey responses may contain sensitive personal information. Data collection, storage, processing, and simulation should follow applicable informed-consent requirements, institutional ethics approvals, and data-protection practices.

AI-generated responses are research simulations, not replacements for actual human responses or definitive representations of an individual's beliefs.

## Acknowledgements

SimTamil is a research project exploring culturally grounded generative AI for the study of human attitudes and social behaviour in Tamil-speaking populations.

---

*SimTamil — Grounding generative AI in individual experiences and cultural context.*

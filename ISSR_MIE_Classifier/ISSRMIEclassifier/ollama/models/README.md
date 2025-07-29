# Ollama Models

This directory contains Ollama model files and related assets. 

# MICclass Model

## Official Response Format

All LLM-based MIE classification outputs must strictly follow this format:

**Prompt Structure:**
```
Classify as MIE or NOT_MIE and provide:
1. Classification (MIE/NOT_MIE)
2. Reasoning
3. Action type (if MIE)
4. Countries involved
5. Fatalities (if any)
```

**Example Response:**
```
1. Classification: MIE
2. Reasoning: This event meets the criteria for an MIE. The U.S. military conducted targeted airstrikes against a non-member state (Syria) utilizing a non-routine, governmentally authorized action – military airstrikes. The targeting of "ISIS positions" constitutes a clear hostile action.
3. Action type: 03 (Military Assistance - Airstrikes)
4. Countries involved: United States, Syria
5. Fatalities: Not specified in the text.
```

All prompts and responses in the system should adhere to this structure for consistency and clarity. 
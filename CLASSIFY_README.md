# GPT-5 Classification Script

This script uses GPT-5.2 to classify phase 2 responses from the exp6_reflection experiment.

## Setup

1. Install dependencies:
```bash
pip install -r requirements_classify.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run the classification script:
```bash
python classify_responses.py
```

## What it does

1. Loads `results_27b/exp6_reflection/exp6_full.json`
2. Filters for cases where `leaked=True` (successful prompt injection)
3. Uses GPT-5.2 to classify each phase 2 response into:
   - **CONFABULATION**: Model fabricates irrelevant reasons
   - **DENIAL**: Model denies mentioning the concept
   - **AWARENESS**: Model realizes something went wrong/was injected
   - **PUZZLEMENT**: Model is confused but doesn't fabricate reasons

4. Saves results to `results_27b/exp6_reflection/exp6_gpt5_classifications.json`

## Output

The output JSON contains:
- Original phase 1 and phase 2 responses
- GPT-5 classification
- Original grade (for comparison)
- Metadata (concept, layer, strength, etc.)

The script also prints summary statistics showing the distribution across categories.

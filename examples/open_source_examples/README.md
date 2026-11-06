# Read first to correctly work with the provided examples


## Open Router
1. Sign up for an Openrouter Account - https://accounts.openrouter.ai/sign-up
2. Create an API key - https://openrouter.ai/keys
3. Add API key to environment - `export OPENROUTER_API_KEY=your key here`
4. Add Openrouter API endpoint to environment - `export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` [See https://openrouter.ai/docs#format for potential updates]

## Perplexity
1. Sign up for an Openrouter Account - https://www.perplexity.ai/
2. Create an API key - https://www.perplexity.ai/pplx-api
3. Add API key to environment - `export PERPLEXITY_API_KEY=your key here`
4. Add Openrouter API endpoint to environment - `export PERPLEXITY_BASE_URL=https://api.perplexity.ai` [See https://docs.perplexity.ai/reference/post_chat_completions for potential updates]

## Runpod
1. Sign up for a Runpod account - https://www.runpod.io/console/signup
2. Add credits, unfortunately no free tier. - https://www.runpod.io/console/user/billing
3. Navigate to templates page[Left selection menu], under `Official` click deploy on `RunPod TheBloke LLMs` template. - https://www.runpod.io/console/templates
4. Navigate to Community Cloud page [Left Selection menu], Click `Deploy` on a GPU with >=16 GB, 1x RTX 4000 Ada SFF works. - https://www.runpod.io/console/gpu-cloud
5. Click `Customize Deployment`, click the `Environment Variables` drop down, Enter the following Key/Values, then click `Set Overrides`, then click `Continue`, and finally `Deploy`.
    - key=MODEL value=TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ
    - key=UI_ARGS value=--n-gpu-layers 100 --threads 1
6. Navigate to Pods[Left selection menu], wait until you see `Connect` button on the Pod you just deployed, click it. Right click `HTTP Service[Port 5000]` and copy the link address. - https://www.runpod.io/console/pods
    - Add Runpod API endpoint to environment - `export RUNPOD_BASE_URL=your-runpod-link/v1` <-- Make sure to add v1 as well
    - Add Runpod API key to environment -  `export RUNPOD_API_KEY="None"` <-- This should be none.
7. When done running, stop instance by clicking the stop icon on the Pod page. - https://www.runpod.io/console/pods
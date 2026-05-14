# Open House Deployment Plan

## What to use on demo day

Use the React frontend as the main screen. It now supports:

- `REACT_APP_API_URL` for a stable backend URL
- demo fallback mode if the backend is down
- visible status message showing whether the live API or fallback is being used

## Recommended path

1. Deploy the frontend in `urtox-ui` to Vercel.
2. Deploy the backend API to Hugging Face Spaces.
3. In Vercel, set:

```bash
REACT_APP_API_URL=https://your-space-name.hf.space
REACT_APP_DEMO_FALLBACK=true
```

4. Test once with the backend running.
5. Test once with the backend stopped, so you know fallback mode works.

## Vercel frontend settings

Use these values when importing `urtox-ui` into Vercel:

```bash
Root Directory: urtox-ui
Framework Preset: Create React App
Build Command: npm run build
Output Directory: build
```

Environment variables:

```bash
REACT_APP_API_URL=https://your-space-name.hf.space
REACT_APP_DEMO_FALLBACK=true
```

## Demo flow

1. Open the Vercel frontend.
2. Click `Safe demo text`, then `Detect Toxic Spans`.
3. Click `Toxic demo text`, then `Detect Toxic Spans`.
4. If the status banner says fallback mode, explain that the frontend is protected against Colab/ngrok downtime.
5. If the status banner says live API connected, explain that the deployed model API is responding.

## Emergency plan

If Hugging Face setup is not finished in time, keep your current Colab/ngrok URL in `REACT_APP_API_URL`. The UI will still show a demo fallback result if Colab disconnects.

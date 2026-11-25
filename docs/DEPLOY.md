# Deploying

## Docker (local)

```bash
docker build -t paper-assistant .          # downloads 50 PDFs and the embedding model while building
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=sk-ant-... paper-assistant
```

Open http://localhost:7860. The build needs internet access to arxiv.org and huggingface.co.
It takes a few minutes: arXiv asks for a pause between downloads, and the model is downloaded once.

The image contains the PDFs and the index, so the running container starts in seconds. The API key
is passed at run time and is never baked into the image.

## Hugging Face Spaces (public demo)

1. Create a new Space: **SDK = Docker**, hardware = CPU basic.
2. Add the Space as a git remote and push this repo to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. Hugging Face needs this block at the very top of `README.md` **in the Space repo**. Add it in a
   commit you push only to the Space, so your GitHub README stays clean:
   ```yaml
   ---
   title: Research Paper Assistant
   emoji: 📄
   sdk: docker
   app_port: 7860
   ---
   ```
4. In the Space: **Settings -> Variables and secrets -> New secret**, name `ANTHROPIC_API_KEY`.
5. **Set a monthly spend cap on that API key in the Anthropic console first.** The Space is public.
6. Wait for the build to finish, then ask a question and a deliberately unanswerable one.

## Cost control

- Default models are Haiku-class (`GENERATION_MODEL`, `JUDGE_MODEL` in `.env`).
- The eval caches every Claude call under `.cache/llm`, so re-running is free.
- Each question sends 5 chunks (about 2k tokens).

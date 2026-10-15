Speed results from the 2026-10-14/15 A6000 run, set aside: another job (make_paraphrase_tsv.py, 21.5 GB,
100% utilization) shared the GPU for the whole run, and the GPU was thermally throttled (92 C, SM clock
210 of 2100 MHz). Decode came out 2x slower than the previous build on the same GPU. Not comparable;
rerun on an idle GPU. The correctness results from the same run (gate.json, ppl.json) stay in results/a6000.

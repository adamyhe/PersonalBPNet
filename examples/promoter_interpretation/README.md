# Promoter interpretation: reference choice for CLIPNET_TF

Scripts for comparing DeepLIFT/SHAP and PISA reference sets for the CLIPNET
9-fold ensemble (`personal_bpnet.clipnet_tf_fast`) at 11 promoters expressed
in lymphoblastoid cells. Each attribution is scored against reference-free in
silico mutagenesis (ISM) of the same target.

| Method | Scored against |
|---|---|
| `deep_lift_shap(target="counts")` | ISM of the counts head |
| `deep_lift_shap(target="shape")` | ISM of `sum_j w_j p_j` |
| `pisa(target="track")` at the main initiation site | ISM of that site's predicted track |
| `pisa(target="profile")` at the same site | ISM of that site's raw profile output |

There are three reference sets, 20 references each:

- `standard`: dinucleotide shuffles (`random_state=47`).
- `quiet`: the 20 lowest-activity shuffles of 500, scored as the PRO-cap atlas's `select_reference_pool` scores them. Ties at the clipped score of 0 are broken by the unclipped sum.
- `mass`: shuffles no louder than the median in counts and 5 bp peak height, ranked by how close their raw profile mass is to the sequence's.

## Inputs

None of these are in the repository.

- **CLIPNET weights:** `fold_1.h5`..`fold_9.h5` from [huggingface.co/adamyhe/clipnet](https://huggingface.co/adamyhe/clipnet):
  ```sh
  for i in $(seq 1 9); do
    curl -sSfL -o clipnet_models/fold_$i.h5 \
      https://huggingface.co/adamyhe/clipnet/resolve/main/fold_$i.h5
  done
  ```
- **Genome:** hg38 FASTA, uncompressed, e.g. UCSC `hg38.fa`.
- **Annotation:** a RefSeq GTF with `gene_name` attributes, e.g. UCSC's `hg38.refGene.gtf.gz`. It is only needed for `observed_tss.py`.
- **Observed PRO-cap:** ENCODE [ENCSR083AMN](https://www.encodeproject.org/experiments/ENCSR083AMN/), primary B cells. ENCODE has no GM12878 PRO-cap, and lymphoblastoid lines are EBV-transformed B cells. You need the four GRCh38 strand signal bigWigs, saved as `<accession>.bigWig`: `ENCFF746KYF`, `ENCFF812MRP` (plus) and `ENCFF408NIK`, `ENCFF074SSA` (minus).

Run with the `tf` extra (`uv run --extra tf python ...`) for `h5py`. `pyBigWig` comes with tangermeme.

## Steps

```sh
# 1. Dominant observed TSS per gene (within 300 bp of its RefSeq starts), with
#    the model's predicted peak as a cross-check -> observed_tss.tsv
python observed_tss.py hg38.refGene.gtf.gz bigwigs/ hg38.fa clipnet_models/ observed_tss.tsv

# 2. Reference selection, DeepLIFT/SHAP, PISA and ISM per promoter -> results/*.npz
python promoters.py clipnet_models/ hg38.fa results/

# 3. Add PISA of the raw profile output (and its ISM) to the results
python pisa_profile_addon.py clipnet_models/ results/

# 4. Agreement tables (stdout, and results/agreement.npy), then figures
python analyze.py results/
python plots.py results/ figures/
```

`promoters.py` has the observed TSSs from step 1 written into its `PROMOTERS`
list. Rerun step 1 and update that list to add promoters. Minus-strand windows
are reverse-complemented, so every promoter reads 5'→3' with its TSS at window
position 500, on the + strand outputs. On an RTX 3080, step 2 takes 2–9 s per
promoter after the models load.

`observed_tss.tsv` is the step 1 output used for the results below. Its
`refseq_used` column holds the RefSeq starts first used to center the windows.
They were wrong by 797 bp (MYC, an alternative upstream start), 185 bp (CD74),
5 bp (NFKBIA), 4 bp (RPLP0) and 1 bp (CD79A). The observed TSS and the model's
predicted peak agree to the base at all 11 promoters.

## Results

Mean Pearson r with ISM over −100..+60, 11 promoters:

| Target | standard | quiet | mass |
|---|---|---|---|
| counts | 0.82 | 0.82 | 0.82 |
| shape | 0.68 | 0.68 | 0.65 |
| PISA, track | 0.59 | 0.53 | 0.57 |
| PISA, raw profile | 0.52 | 0.51 | 0.51 |

- **Counts and shape:** the reference set barely matters for either.
- **PISA (track):** it is unstable at the strongest promoters, where the best reference set changes from one promoter to the next:

  | Promoter | standard | quiet | mass |
  |---|---|---|---|
  | IRF4 | 0.70 | −0.02 | 0.19 |
  | JUNB | 0.23 | 0.20 | 0.86 |

  At these promoters the sequence's raw profile mass is 3–6. Even the mass-matched references average 18–20, and the standard and quiet ones 25–32, so the track's normalization couples each output to the total mass.
- **ISM is a yardstick, not ground truth:** at ACTB almost every mutation lowers the shape target slightly, a floor that DeepLIFT doesn't reproduce.

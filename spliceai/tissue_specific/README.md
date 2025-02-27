## Using long-read data from ENCODE
```bash
samtools index ENCFF932MJL.bam

bcftools mpileup -Ou -f GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta ENCFF932MJL.bam | bcftools call -mv -Oz -o ENCSR792OIJ_variants.vcf.gz

bcftools view -v snps -m2 -M2 HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz -Oz -o GM12878_SNPs_biallelic.vcf.gz

bcftools index GM12878_SNPs_biallelic.vcf.gz

bcftools consensus -f GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -H 1 GM12878_SNPs_biallelic.vcf.gz > GM12878.fasta
```x

# Using GIAB NA12878 variants
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/latest/GRCh38/
```bash
bcftools view -v snps -m2 -M2 HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz -Oz -o GM12878_SNPs_biallelic.vcf.gz

bcftools index GM12878_SNPs_biallelic.vcf.gz

bcftools consensus -f GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -H 1 GM12878_SNPs_biallelic.vcf.gz > GM12878.fasta
```

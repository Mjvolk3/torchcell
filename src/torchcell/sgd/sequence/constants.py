# can get from [print(chr.chrom) for chr in self.db.features_of_type("chromosome")]
# or genome.db.fatures_of_type("chromosome")
CHROMOSOMES = [
    "chrmt",  # We put MT at 0, because it is circular, and this preserves arabic to roman
    "chrI",
    "chrII",
    "chrIII",
    "chrIV",
    "chrV",
    "chrVI",
    "chrVII",
    "chrVIII",
    "chrIX",
    "chrX",
    "chrXI",
    "chrXII",
    "chrXIII",
    "chrXIV",
    "chrXV",
    "chrXVI",
]

if __name__ == "__main__":
    pass

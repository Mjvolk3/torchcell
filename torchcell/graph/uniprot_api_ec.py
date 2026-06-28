# torchcell/multidigraph/uniprot_api_ec.py
# [[torchcell.multidigraph.uniprot_api_ec]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/multidigraph/uniprot_api_ec.py
# Test file: torchcell/multidigraph/test_uniprot_api_ec.py
"""Fetch EC numbers for proteins from the UniProt REST API."""

import requests

# Copy paste from the S288C gff file
# Parent=YBL105C_id001,YBL105C_id002;Name=YBL105C_CDS;orf_classification=Verified;protein_id=UniProtKB:P24583


def get_uniprot_ec(uniprot_id: str) -> str | None:
    """Return the first EC number listed in the UniProt text record for a protein."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)
    for line in response.text.split("\n"):
        if "EC=" in line:
            return line.split("EC=")[1].split(";")[0].strip()
    return None


ec_number = get_uniprot_ec("P24583")
print("EC Number:", ec_number)

# Should output EC Number: 2.7.11.13
# This information is contained in the gbff file also.


def main() -> None:
    """Entry point placeholder for running this module as a script."""
    pass


if __name__ == "__main__":
    main()

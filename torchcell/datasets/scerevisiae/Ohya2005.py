import requests
from pathlib import Path
from typing import Tuple, Optional


def download_yeast_data(
    url: str = "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=mt4718data.tsv",
    output_path: Optional[Path] = None,
    timeout: int = 60,
) -> Tuple[bool, Optional[Path]]:
    """
    Download yeast data mimicking Safari browser behavior.

    Args:
        url: URL to download the data from
        output_path: Path to save the downloaded data. If None, saves to current directory
        timeout: Timeout in seconds for the request

    Returns:
        Tuple of (success_status, file_path_if_downloaded)
    """
    if output_path is None:
        output_path = Path("mt4718data.tsv")

    # Safari user agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.0 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    try:
        print(f"Attempting to download from {url} with Safari user-agent...")

        # Create a session to maintain cookies
        session = requests.Session()

        # Try to access the main page first to get cookies
        main_url = "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php"
        session.get(main_url, headers=headers, timeout=timeout)

        # Now download the file
        response = session.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

        # Save the file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = output_path.stat().st_size
        if file_size > 0:
            print(
                f"Download complete. File saved to {output_path} ({file_size/1024/1024:.2f} MB)"
            )
            return True, output_path
        else:
            print("Downloaded file is empty.")
            output_path.unlink(missing_ok=True)  # Delete empty file
            return False, None

    except requests.RequestException as e:
        print(f"Error downloading resource: {e}")
        return False, None


if __name__ == "__main__":
    success, file_path = download_yeast_data()
    if not success:
        print("Failed to download with Safari user-agent.")
        print("You may need to manually download the file using Safari browser.")

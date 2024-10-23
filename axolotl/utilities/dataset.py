import httpx
import os

async def download_dataset(s3_url: str, job_id: str) -> str:
    
    destination_path = f"/workspace/axolotl/data/{job_id}.json"

    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        async with httpx.AsyncClient() as client:
            response = await client.get(s3_url)
            response.raise_for_status()

            with open(destination_path, "wb") as file:
                file.write(response.content)

        return destination_path

    except Exception as e:
        raise Exception(f"Error downloading dataset: {str(e)}")
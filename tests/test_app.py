import asyncio
import httpx


async def main():
    # This matches your FastAPI URL and port
    url = "http://127.0.0.1:8000/api/ai/ask"

    # Example question that might trigger the SQL fallback
    payload = {"question": "How many listings are in New York?"}

    print(f"Sending request to {url}...")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            print("\nResponse Status:", response.status_code)
            print("Response Body:\n", response.json())

        except httpx.HTTPStatusError as e:
            print(f"Error response: {e.response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

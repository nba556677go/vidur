import pandas as pd

def save_prompts_for_vllm(parquet_path, output_path, num_prompts=100):
    """
    Reads prompts from a Parquet file and saves them to a text file,
    with one prompt per line, for use with vllm.

    Args:
        parquet_path (str): The path to the source .parquet file.
        output_path (str): The path for the output .txt file.
        num_prompts (int): The number of prompts to extract.
    """
    print(f"Reading prompts from '{parquet_path}'...")
    try:
        df = pd.read_parquet(parquet_path)

        if 'conversation' not in df.columns:
            print("Error: A 'conversation' column was not found in the Parquet file.")
            return

        prompts = []
        # Iterate through the conversation column
        for conv in df['conversation'].head(num_prompts):
            # --- FIX IS HERE ---
            # Check explicitly if the conversation list is valid and not empty
            if conv is not None and len(conv) > 0 and isinstance(conv[0], dict) and 'content' in conv[0]:
                # Get the content of the first turn
                content = conv[0]['content']
                # Replace newline characters within a single prompt with a space
                # to ensure each prompt stays on one line for vllm.
                cleaned_prompt = content.replace('\n', ' ')
                prompts.append(cleaned_prompt)
        
        if not prompts:
            print("No valid prompts were extracted. The file will not be created.")
            return

        # Write the collected prompts to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')

        print(f"Successfully saved {len(prompts)} prompts to '{output_path}'.")
        print("The file is ready to be used with the --prompt-file argument in vllm.")

    except FileNotFoundError:
        print(f"Error: The file was not found at {parquet_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # The name of one of your Parquet files
    source_parquet_file = 'lmsys-chat-1m/data/train-00000-of-00006-4feeb3f83346a0e9.parquet'
    num_prompts = 150
    # The desired name for your vllm prompt file
    output_prompt_file = f'prompt_numprompts{num_prompts}.txt'
    
    # --- Execution ---
    save_prompts_for_vllm(source_parquet_file, output_prompt_file, num_prompts=num_prompts)
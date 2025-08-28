import pandas as pd
from transformers import AutoTokenizer

def process_and_save_prompts(
    parquet_path, 
    output_path, 
    model_name, 
    target_length, 
    num_prompts
    ):
    """
    Reads prompts, adjusts them to a specific token length, and saves them
    in a format ready for vllm's --prompt-file argument.

    Args:
        parquet_path (str): Path to the source .parquet file.
        output_path (str): Path for the output .txt file.
        model_name (str): The Hugging Face model identifier for the tokenizer.
        target_length (int): The desired token length for each prompt.
        num_prompts (int): The number of prompts to process.
    """
    print("Loading tokenizer...")
    try:
        # Load the tokenizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for '{model_name}': {e}")
        print("Please ensure you have an internet connection and the model name is correct.")
        return

    print(f"Reading and processing prompts from '{parquet_path}'...")
    try:
        df = pd.read_parquet(parquet_path)
        if 'conversation' not in df.columns:
            print("Error: 'conversation' column not found.")
            return

        processed_prompts = []
        source_prompts = df['conversation'].head(num_prompts)

        for conv in source_prompts:
            if conv is not None and len(conv) > 0 and isinstance(conv[0], dict) and 'content' in conv[0]:
                original_prompt = conv[0]['content']
                
                # Tokenize the original prompt
                input_ids = tokenizer.encode(original_prompt)
                
                # Adjust token length
                if len(input_ids) > target_length:
                    # --- TRUNCATE ---
                    final_ids = input_ids[:target_length]
                elif len(input_ids) < target_length:
                    # --- EXTEND ---
                    # Repeat the tokens until the target length is met
                    multiplied_ids = input_ids * (target_length // len(input_ids) + 1)
                    final_ids = multiplied_ids[:target_length]
                else:
                    # Length is already correct
                    final_ids = input_ids
                
                # Decode back to a string
                final_prompt = tokenizer.decode(final_ids, skip_special_tokens=False)
                
                # Clean up any newlines to ensure one prompt per line
                cleaned_prompt = final_prompt.replace('\n', ' ').strip()
                processed_prompts.append(cleaned_prompt)

        if not processed_prompts:
            print("No valid prompts were extracted. The file will not be created.")
            return

        # Write the final prompts to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for prompt in processed_prompts:
                f.write(prompt + '\n')

        print("-" * 50)
        print(f"Successfully saved {len(processed_prompts)} prompts to '{output_path}'.")
        print(f"Each prompt has been adjusted to ~{target_length} tokens.")
        print("File is ready for use with vllm.")

    except FileNotFoundError:
        print(f"Error: The file was not found at {parquet_path}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # 1. UPDATED: The full path to your source file
    source_parquet_file = '/home/ec2-user/s3-local/benchmarks/llm/prompts/lmsys-chat-1m/data/train-00000-of-00006-4feeb3f83346a0e9.parquet'
    
   
    
    # 3. Target token length for each prompt
    TOKEN_LENGTH = 152


    num_prompts=2000
    # 2. The desired name for your vllm prompt file
    output_prompt_file = f'prompt_extend_{TOKEN_LENGTH}_numprompts{num_prompts}.txt'
    
    # 4. The model you are using (for the tokenizer).
    # NOTE: "qwen-2.5b" is not a recognized model on Hugging Face as of late 2024.
    # Using a valid Qwen model like "Qwen/Qwen1.5-1.8B-Chat" instead. 
    # Please change this if you have a different specific model name.
    TOKENIZER_MODEL = "Qwen/Qwen2.5-1.5B"
    #TOKENIZER_MODEL = "meta-llama/Meta-Llama-3-8B"
    
    # --- Execution ---
    process_and_save_prompts(
        parquet_path=source_parquet_file,
        output_path=output_prompt_file,
        model_name=TOKENIZER_MODEL,
        target_length=TOKEN_LENGTH,
        num_prompts=num_prompts
    )
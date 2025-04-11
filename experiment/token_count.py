from transformers.models.auto.tokenization_auto import AutoTokenizer
import click
from conf import TORCH_BASE
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pathlib import Path
import pickle

# Define the path to the font file
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TimesSong.ttf')

# Create a FontProperties object for the Chinese font
chinese_font = FontProperties(fname=FONT_PATH)

# Configure matplotlib to use the font for all text elements
plt.rcParams.update({
    "font.size": 10,
    "mathtext.fontset": 'stix',
    'axes.unicode_minus': False,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.labelcolor": "black",
    "axes.labelpad": 10,
})

def count_tokens(text, model_name="deepseek-ai/DeepSeek-V3"):
    """
    Count the number of tokens in the given text using the DeepSeek tokenizer.
    
    Args:
        text (str): The text to count tokens for
        model_name (str): The name of the model to use for tokenization
        
    Returns:
        int: The number of tokens in the text
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Return the number of tokens
    return len(tokens)

@click.command()
@click.option('--model', '-m', default="deepseek-ai/DeepSeek-V3", 
              help='Model name to use for tokenization')
@click.option('--output', '-o', default="token_distribution.png",
              help='Output file for the visualization')
@click.option('--cache', '-c', default="token_counts_cache.pkl",
              help='Cache file for token counts')
@click.option('--force', '-f', is_flag=True,
              help='Force recalculation even if cache exists')
def main(model, output, cache, force):
    """Count tokens in text using DeepSeek tokenizer and visualize the distribution."""
    inductor_path = TORCH_BASE / "torch" / "_inductor"
    
    # Check if cache exists and load it if available
    token_counts = []
    file_paths = []
    
    if os.path.exists(cache) and not force:
        click.echo(f"Loading cached results from {cache}")
        try:
            with open(cache, 'rb') as f:
                cached_data = pickle.load(f)
                token_counts = cached_data['token_counts']
                file_paths = cached_data['file_paths']
                click.echo(f"Loaded {len(token_counts)} cached results")
        except Exception as e:
            click.echo(f"Error loading cache: {e}")
            click.echo("Will recalculate token counts")
    
    # If cache doesn't exist or force flag is set, calculate token counts
    if not token_counts or force:
        click.echo("Calculating token counts...")
        token_counts = []
        file_paths = []
        
        files = list(inductor_path.glob("**/*.py"))
        click.echo(f"Found {len(files)} Python files to analyze")
        
        for i, file in enumerate(files):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    input_text = f.read()
                    token_count = count_tokens(input_text, model)
                    token_counts.append(token_count)
                    file_paths.append(str(file))
                    
                    if (i + 1) % 10 == 0:
                        click.echo(f"Processed {i + 1}/{len(files)} files")
            except Exception as e:
                click.echo(f"Error processing {file}: {e}")
        
        # Save results to cache
        click.echo(f"Saving results to cache {cache}")
        with open(cache, 'wb') as f:
            pickle.dump({'token_counts': token_counts, 'file_paths': file_paths}, f)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Histogram of token counts
    plt.subplot(1, 2, 1)
    plt.hist(token_counts, bins=30, alpha=0.7, color='skyblue')
    plt.title('token数量分布', fontproperties=chinese_font, fontsize=plt.rcParams["axes.titlesize"])
    plt.xlabel('token数量', fontproperties=chinese_font, fontsize=plt.rcParams["axes.labelsize"])
    plt.ylabel('频率', fontproperties=chinese_font)
    
    # Top 10 files by token count
    plt.subplot(1, 2, 2)
    top_indices = np.argsort(token_counts)[-10:]
    top_counts = [token_counts[i] for i in top_indices]
    top_files = [os.path.basename(file_paths[i]) for i in top_indices]
    plt.barh(top_files, top_counts, color='lightgreen')
    plt.title('token数量最多的10个文件', fontproperties=chinese_font, fontsize=plt.rcParams["axes.titlesize"])
    plt.xlabel('token数量', fontproperties=chinese_font, fontsize=plt.rcParams["axes.labelsize"])
    
    # Set font for y-axis labels (file names)
    plt.yticks(fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Visualization saved to {output}")
    
    # Print summary statistics
    click.echo(f"\nSummary Statistics:")
    click.echo(f"Total files analyzed: {len(token_counts)}")
    click.echo(f"Mean token count: {np.mean(token_counts):.2f}")
    click.echo(f"Median token count: {np.median(token_counts):.2f}")
    click.echo(f"Min token count: {min(token_counts)}")
    click.echo(f"Max token count: {max(token_counts)}")
    click.echo(f"Standard deviation: {np.std(token_counts):.2f}")

if __name__ == "__main__":
    main()

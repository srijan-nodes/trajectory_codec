import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading ablation_matrix.csv...")
    df = pd.read_csv("ablation_matrix.csv")
    
    # 1. Scatter Plot: FPS vs Reduction
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Unweighted_Reduction_Percent', y='FPS', alpha=0.5, color='blue', edgecolor='w')
    plt.title("Ablation Study: Encoding FPS vs Data Reduction", fontsize=14, fontweight='bold')
    plt.xlabel("Unweighted Reduction % [Higher is Better]", fontsize=12)
    plt.ylabel("Encoding Speed (FPS) [Higher is Better]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("final_commit/ablation_scatter.png", dpi=150)
    plt.close()
    
    # 2. Bar Chart: Top 10 Configurations
    top10 = df.nlargest(10, 'Unweighted_Reduction_Percent').copy()
    # Format names for readability
    def short_name(name):
        active = [n.replace('+m', '') for n in name.split() if '+' in n]
        if not active: return 'Base (Raw)'
        if len(active) > 4: return f"{len(active)} Modes Active"
        return '+'.join(active)
        
    top10['Short_Name'] = top10['Permutation'].apply(short_name)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=top10, x='Unweighted_Reduction_Percent', y='Short_Name', palette='viridis')
    plt.title("Top 10 Algorithmic Configurations by Data Reduction", fontsize=14, fontweight='bold')
    plt.xlabel("Unweighted Reduction %", fontsize=12)
    plt.ylabel("Active Modules", fontsize=12)
    
    # Add value labels
    for i, v in enumerate(top10['Unweighted_Reduction_Percent']):
        ax.text(v - 0.05, i, f"{v:.2f}%", color='white', va='center', fontweight='bold', ha='right')
        
    plt.tight_layout()
    plt.savefig("final_commit/ablation_top10.png", dpi=150)
    plt.close()
    
    print("Successfully generated final_commit/ablation_scatter.png and final_commit/ablation_top10.png!")

if __name__ == "__main__":
    main()

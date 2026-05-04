import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, facecolor='#EAEAEA', fontsize=10):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor='black', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontfamily='sans-serif', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2, label, ha='center', va='bottom', fontsize=9, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'DermoViT Architecture', ha='center', va='center', fontsize=16, fontweight='bold')

# Input
draw_box(ax, 3.5, 10.5, 3, 0.8, 'Raw Dermoscopy Image\n(B, 3, 224, 224)', facecolor='#E3F2FD')

# Streams
draw_box(ax, 1, 8.5, 3.5, 1, 'Local Stream\nEfficientNet-B2\n(B, 1408, 7, 7)', facecolor='#FFCDD2')
draw_box(ax, 5.5, 8.5, 3.5, 1, 'Global Stream\nDeiT-Small\n(B, 197, 384)', facecolor='#C8E6C9')

draw_arrow(ax, 5, 10.4, 2.75, 9.6)
draw_arrow(ax, 5, 10.4, 7.25, 9.6)

# Projections
draw_box(ax, 1, 6.8, 3.5, 0.8, 'Flatten & Project\n(B, 49, 512)', facecolor='#FFF9C4')
draw_box(ax, 5.5, 6.8, 3.5, 0.8, 'Linear Project\n(B, 197, 512)', facecolor='#FFF9C4')

draw_arrow(ax, 2.75, 8.4, 2.75, 7.7)
draw_arrow(ax, 7.25, 8.4, 7.25, 7.7)

# ACAG Block
# Draw large bounding box for ACAG
rect_acag = patches.Rectangle((0.5, 3.5), 9, 2.8, linewidth=1.5, edgecolor='gray', linestyle='--', facecolor='none')
ax.add_patch(rect_acag)
ax.text(0.7, 6.0, 'Asymmetry-Aware Cross-Attention Gate (ACAG)', ha='left', va='center', fontsize=11, fontweight='bold', color='gray')

draw_box(ax, 1, 4.8, 3.5, 0.8, 'Asymmetry Residual\nA = |F - F_h| + |F - F_v|', facecolor='#D1C4E9')
draw_box(ax, 1, 3.7, 3.5, 0.8, 'Query Bias\nQ = W_q(CNN) + λ·A', facecolor='#D1C4E9')
draw_box(ax, 5.5, 4.2, 3.5, 1.2, 'Cross-Attention\nOut = softmax(Q·K^T / √d)·V', facecolor='#D1C4E9')

draw_arrow(ax, 2.75, 6.7, 2.75, 5.7)
draw_arrow(ax, 2.75, 4.7, 2.75, 4.6)
draw_arrow(ax, 4.6, 4.1, 5.4, 4.6)  # Q to cross-attn
draw_arrow(ax, 7.25, 6.7, 7.25, 5.5, 'K, V') # ViT to cross-attn

# Metadata & FiLM
rect_film = patches.Rectangle((0.5, 1.5), 9, 1.7, linewidth=1.5, edgecolor='gray', linestyle='--', facecolor='none')
ax.add_patch(rect_film)
ax.text(0.7, 2.9, 'Clinical Metadata Conditioning', ha='left', va='center', fontsize=11, fontweight='bold', color='gray')

draw_box(ax, 1, 1.8, 3.5, 0.8, 'Tabular Metadata\nAge, Sex, Site (B, 30)', facecolor='#FFECB3')
draw_box(ax, 5.5, 1.8, 3.5, 0.8, 'FiLM Layer\nγ(m) ⊙ x + β(m)', facecolor='#B3E5FC')

draw_arrow(ax, 4.6, 2.2, 5.4, 2.2, 'MLP γ, β')
draw_arrow(ax, 7.25, 4.1, 7.25, 2.7, 'Attended Features (B, 512)')

# Head
draw_box(ax, 3.5, 0.2, 3, 0.8, 'Classification Head\nLogit → Sigmoid', facecolor='#E0E0E0')
draw_arrow(ax, 7.25, 1.7, 5.0, 1.1)

plt.tight_layout()
plt.savefig('DermoViT_Architecture.png', dpi=300, bbox_inches='tight')
print("Successfully generated DermoViT_Architecture.png")

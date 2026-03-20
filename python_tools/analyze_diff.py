"""Analyze difference between rendered 3D model and original image"""
import cv2
import numpy as np

# Load images
original = cv2.imread("Image0.png")
rendered = cv2.imread("rotation_results_cpp/rendered_textured.png")
overlay50 = cv2.imread("rotation_results_cpp/overlay_50.png")

print(f"Original: {original.shape}")
print(f"Rendered: {rendered.shape}")

# Create mask from rendered (non-black pixels)
rend_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
rend_mask = (rend_gray > 0).astype(np.uint8)

# Create mask from original (using depth-based mask)
# Re-create the mask as in the C++ pipeline
depth = cv2.imread("Image0_depth.png")
b, g, r = cv2.split(depth)
depth_val = g.astype(np.float32) - b.astype(np.float32)
mask = ((depth_val > 5) | (g > 10)).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((labels == largest) * 255).astype(np.uint8)

# Fill holes
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_filled = np.zeros_like(mask)
cv2.drawContours(mask_filled, contours, -1, 255, cv2.FILLED)

print(f"Mask filled pixels: {np.count_nonzero(mask_filled)}")
print(f"Rendered pixels: {np.count_nonzero(rend_mask)}")

# IoU
intersection = np.count_nonzero(mask_filled & (rend_mask * 255))
union = np.count_nonzero(mask_filled | (rend_mask * 255))
print(f"IoU: {intersection/union*100:.2f}%")

# Mismatch analysis
orig_only = (mask_filled > 0) & (rend_mask == 0)  # In original but not rendered
rend_only = (rend_mask > 0) & (mask_filled == 0)   # In rendered but not original

print(f"\nMismatch areas:")
print(f"  Original only (missing in render): {np.count_nonzero(orig_only)} pixels")
print(f"  Rendered only (extra in render): {np.count_nonzero(rend_only)} pixels")

# Find bounding boxes of mismatch regions
# Original only (what's missing from the render)
if np.any(orig_only):
    ys, xs = np.where(orig_only)
    print(f"  Original-only region: x=[{xs.min()}-{xs.max()}], y=[{ys.min()}-{ys.max()}]")

    # Split into connected components to understand separate mismatch regions
    orig_only_u8 = orig_only.astype(np.uint8) * 255
    n_comp, comp_labels, comp_stats, comp_centroids = cv2.connectedComponentsWithStats(orig_only_u8)
    print(f"  Number of separate mismatch regions: {n_comp - 1}")
    # Show top 5 largest
    areas = [(i, comp_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_comp)]
    areas.sort(key=lambda x: -x[1])
    for idx, (comp_id, area) in enumerate(areas[:5]):
        cx = comp_centroids[comp_id][0]
        cy = comp_centroids[comp_id][1]
        x0 = comp_stats[comp_id, cv2.CC_STAT_LEFT]
        y0 = comp_stats[comp_id, cv2.CC_STAT_TOP]
        w = comp_stats[comp_id, cv2.CC_STAT_WIDTH]
        h = comp_stats[comp_id, cv2.CC_STAT_HEIGHT]
        print(f"    Region {idx+1}: {area} pixels at center ({cx:.0f},{cy:.0f}), bbox=({x0},{y0},{x0+w},{y0+h})")

if np.any(rend_only):
    rend_only_u8 = rend_only.astype(np.uint8) * 255
    n_comp2, _, comp_stats2, comp_centroids2 = cv2.connectedComponentsWithStats(rend_only_u8)
    areas2 = [(i, comp_stats2[i, cv2.CC_STAT_AREA]) for i in range(1, n_comp2)]
    areas2.sort(key=lambda x: -x[1])
    print(f"\n  Rendered-only regions: {n_comp2-1} separate regions")
    for idx, (comp_id, area) in enumerate(areas2[:5]):
        cx = comp_centroids2[comp_id][0]
        cy = comp_centroids2[comp_id][1]
        print(f"    Region {idx+1}: {area} pixels at center ({cx:.0f},{cy:.0f})")

# ORB feature analysis
print("\n=== ORB Feature Analysis ===")
orb = cv2.ORB_create(3000)

# Get bbox of overlap region
overlap = (mask_filled > 0) & (rend_mask > 0)
ys, xs = np.where(overlap)
x0, y0, x1, y1 = max(0, xs.min()-10), max(0, ys.min()-10), min(original.shape[1]-1, xs.max()+10), min(original.shape[0]-1, ys.max()+10)
roi = slice(y0, y1+1), slice(x0, x1+1)

crop_orig = original[roi]
crop_rend = rendered[roi]
crop_omask = mask_filled[roi]
crop_rmask = (rend_mask * 255)[roi]

kp1, des1 = orb.detectAndCompute(crop_orig, crop_omask)
kp2, des2 = orb.detectAndCompute(crop_rend, crop_rmask)
print(f"Keypoints: original={len(kp1)}, rendered={len(kp2)}")

if des1 is not None and des2 is not None and len(kp1) >= 5 and len(kp2) >= 5:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]
    print(f"Total matches: {len(matches)}, Good (dist<50): {len(good)}")

    if good:
        offsets = []
        for m in good:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            offsets.append(np.sqrt(dx*dx + dy*dy))
        offsets = np.array(offsets)
        print(f"Good match offsets: mean={offsets.mean():.1f}, median={np.median(offsets):.1f}, max={offsets.max():.1f}")
        score = len(good) / (1 + offsets.mean())
        print(f"Feature score: {score:.2f}")

    # Draw matches
    match_img = cv2.drawMatches(crop_orig, kp1, crop_rend, kp2,
                                 sorted(matches, key=lambda x: x.distance)[:50],
                                 None, flags=2)
    cv2.imwrite("rotation_results_cpp/feature_matches.png", match_img)
    print("Saved feature_matches.png")

# Create difference visualization
diff_vis = original.copy()
# Red = in original only (missing from render)
diff_vis[orig_only] = [0, 0, 255]
# Blue = in render only (extra)
diff_vis[rend_only] = [255, 0, 0]
cv2.imwrite("rotation_results_cpp/mismatch_regions.png", diff_vis)
print("\nSaved mismatch_regions.png (Red=missing in render, Blue=extra in render)")

# Also save zoomed comparison of problem areas
# Create side-by-side of original vs rendered in the model region
bb = cv2.boundingRect(mask_filled)
x, y, w, h = bb
margin = 20
x0 = max(0, x - margin)
y0 = max(0, y - margin)
x1 = min(original.shape[1], x + w + margin)
y1 = min(original.shape[0], y + h + margin)

crop_o = original[y0:y1, x0:x1]
crop_r = rendered[y0:y1, x0:x1]

# Compute per-pixel difference in the overlap region
overlap_crop = overlap[y0:y1, x0:x1]
diff = np.zeros_like(crop_o)
diff[overlap_crop] = cv2.absdiff(crop_o, crop_r)[overlap_crop]

# Amplify difference for visibility
diff_amp = np.clip(diff.astype(np.float32) * 3, 0, 255).astype(np.uint8)

comparison = np.hstack([crop_o, crop_r, diff_amp])
cv2.imwrite("rotation_results_cpp/comparison.png", comparison)
print("Saved comparison.png (Original | Rendered | Difference x3)")

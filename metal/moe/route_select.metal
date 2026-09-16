uint token = thread_position_in_grid.x;
if (token >= count[0]) return;

int best_ids[7];
float best_scores[7];
for (uint j = 0; j < 7; ++j) {
    best_ids[j] = -1;
    best_scores[j] = 0.0f;
}

bool valid = true;
uint base = token * 384;
for (int expert = 0; expert < 384; ++expert) {
    float raw = scores[base + uint(expert)];
    float corrected = raw + bias[expert];
    valid = valid && isfinite(raw) && raw >= 0.0f && isfinite(corrected);
    int insert = 7;
    for (int j = 0; j < 7; ++j) {
        if (best_ids[j] < 0 || corrected > best_scores[j] ||
            (corrected == best_scores[j] && expert < best_ids[j])) {
            insert = j;
            break;
        }
    }
    if (insert < 7) {
        for (int j = 6; j > insert; --j) {
            best_ids[j] = best_ids[j - 1];
            best_scores[j] = best_scores[j - 1];
        }
        best_ids[insert] = expert;
        best_scores[insert] = corrected;
    }
}

for (uint j = 0; j < 7; ++j) ids[token * 7 + j] = uint(best_ids[j]);
for (uint j = 0; j < 6; ++j) {
    selected_raw[token * 6 + j] = scores[base + uint(best_ids[j])];
    lhs_ids[token * 6 + j] = token;
}
uint slots[6] = {0, 1, 2, 3, 4, 5};
for (uint i = 1; i < 6; ++i) {
    uint slot = slots[i];
    uint j = i;
    while (j > 0 && best_ids[slots[j - 1]] > best_ids[slot]) {
        slots[j] = slots[j - 1];
        --j;
    }
    slots[j] = slot;
}
for (uint j = 0; j < 6; ++j) reduction_slots[token * 6 + j] = slots[j];
boundary[token * 2] = best_scores[5];
boundary[token * 2 + 1] = best_scores[6];
errors[token] = valid ? uchar(0) : uchar(1);

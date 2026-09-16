uint element = thread_position_in_grid.x;
uint total = params[0] * 5120;
if (element >= total) return;
uint token = element / 5120;
uint column = element - token * 5120;
float sum = 0.0f;
for (uint rank = 0; rank < 6; ++rank) {
    uint assignment = reduction_slots[token * 6 + rank];
    sum += float(weighted[assignment * 5120 + column]);
}
accumulated[element] = sum;

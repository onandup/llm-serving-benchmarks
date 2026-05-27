from dataclasses import dataclass
import random


@dataclass
class AllocationResult:
    allocator_type: str
    total_blocks: int
    successful_allocations: int
    failed_allocations: int
    fragmentation_events: int
    final_free_blocks: int
    peak_used_blocks: int


class ContiguousAllocator:
    def __init__(self, total_blocks: int):
        self.blocks = [None] * total_blocks
        self.peak_used_blocks = 0
        self.fragmentation_events = 0

    def allocate(self, request_id: int, blocks_needed: int) -> bool:
        start = self._find_contiguous_range(blocks_needed)

        if start is None:
            if self.free_blocks() >= blocks_needed:
                self.fragmentation_events += 1
            return False

        for i in range(start, start + blocks_needed):
            self.blocks[i] = request_id

        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks())
        return True

    def release(self, request_id: int):
        for i, owner in enumerate(self.blocks):
            if owner == request_id:
                self.blocks[i] = None

    def _find_contiguous_range(self, blocks_needed: int):
        current_start = None
        current_len = 0

        for i, owner in enumerate(self.blocks):
            if owner is None:
                if current_start is None:
                    current_start = i
                current_len += 1

                if current_len >= blocks_needed:
                    return current_start
            else:
                current_start = None
                current_len = 0

        return None

    def free_blocks(self) -> int:
        return sum(1 for b in self.blocks if b is None)

    def used_blocks(self) -> int:
        return len(self.blocks) - self.free_blocks()


class PagedAllocator:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_block_ids = set(range(total_blocks))
        self.allocations = {}
        self.peak_used_blocks = 0

    def allocate(self, request_id: int, blocks_needed: int) -> bool:
        if len(self.free_block_ids) < blocks_needed:
            return False

        allocated = set(list(self.free_block_ids)[:blocks_needed])
        self.free_block_ids -= allocated
        self.allocations[request_id] = allocated

        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks())
        return True

    def release(self, request_id: int):
        allocated = self.allocations.pop(request_id, set())
        self.free_block_ids |= allocated

    def free_blocks(self) -> int:
        return len(self.free_block_ids)

    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks()


def run_simulation(total_blocks: int, allocator_type: str, seed: int = 42):
    random.seed(seed)

    if allocator_type == "contiguous":
        allocator = ContiguousAllocator(total_blocks)
    elif allocator_type == "paged":
        allocator = PagedAllocator(total_blocks)
    else:
        raise ValueError(f"Unknown allocator type: {allocator_type}")

    active_requests = {}
    successful_allocations = 0
    failed_allocations = 0

    request_id = 0

    for step in range(1000):
        # Randomly release some active requests to create holes.
        if active_requests and random.random() < 0.45:
            release_id = random.choice(list(active_requests.keys()))
            allocator.release(release_id)
            active_requests.pop(release_id)

        # Randomly allocate new requests of different sizes.
        if random.random() < 0.75:
            blocks_needed = random.choice([1, 2, 4, 8, 16])

            success = allocator.allocate(request_id, blocks_needed)

            if success:
                active_requests[request_id] = blocks_needed
                successful_allocations += 1
            else:
                failed_allocations += 1

            request_id += 1

    return AllocationResult(
        allocator_type=allocator_type,
        total_blocks=total_blocks,
        successful_allocations=successful_allocations,
        failed_allocations=failed_allocations,
        fragmentation_events=getattr(allocator, "fragmentation_events", 0),
        final_free_blocks=allocator.free_blocks(),
        peak_used_blocks=allocator.peak_used_blocks,
    )


if __name__ == "__main__":
    for total_blocks in [64, 128, 256, 512]:
        for allocator_type in ["contiguous", "paged"]:
            print(run_simulation(total_blocks, allocator_type))
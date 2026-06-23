import numpy as np


# COCO 17-joint skeleton edges
# Each tuple: (joint_a, joint_b)
COCO_EDGES = [
    (0, 1), (0, 2),         # nose → eyes
    (1, 3), (2, 4),         # eyes → ears
    (0, 5), (0, 6),         # nose → shoulders (via neck approximation)
    (5, 6),                 # shoulders
    (5, 7), (7, 9),         # left arm
    (6, 8), (8, 10),        # right arm
    (5, 11), (6, 12),       # torso sides
    (11, 12),               # hips
    (11, 13), (13, 15),     # left leg
    (12, 14), (14, 16),     # right leg
]

JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

NUM_JOINTS = 17


class SkeletonGraph:
    def __init__(self, strategy='spatial', max_hop=1):
        self.num_joints = NUM_JOINTS
        self.edges = COCO_EDGES
        self.strategy = strategy
        self.max_hop = max_hop
        self.A = self._build_adjacency()

    def _hop_distance(self):
        dist = np.full((self.num_joints, self.num_joints), np.inf)
        np.fill_diagonal(dist, 0)
        for i, j in self.edges:
            dist[i, j] = 1
            dist[j, i] = 1
        for k in range(self.num_joints):
            for i in range(self.num_joints):
                for j in range(self.num_joints):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def _build_adjacency(self):
        dist = self._hop_distance()
        if self.strategy == 'uniform':
            A = (dist <= self.max_hop).astype(float)
            A = self._normalize(A)
            return A[np.newaxis]  # (1, V, V)

        elif self.strategy == 'spatial':
            # 3 subsets: self, centripetal (toward center), centrifugal (away)
            # Center joint = midpoint of hips (11 and 12) → use joint 11 as proxy
            center = 11
            A = []
            for hop in range(self.max_hop + 1):
                a_root = np.zeros((self.num_joints, self.num_joints))
                a_close = np.zeros((self.num_joints, self.num_joints))
                a_far = np.zeros((self.num_joints, self.num_joints))
                for i in range(self.num_joints):
                    for j in range(self.num_joints):
                        if dist[i, j] == hop:
                            if dist[j, center] == dist[i, center]:
                                a_root[i, j] = 1
                            elif dist[j, center] < dist[i, center]:
                                a_close[i, j] = 1
                            else:
                                a_far[i, j] = 1
                if hop == 0:
                    A.append(self._normalize(a_root))
                else:
                    A.append(self._normalize(a_root))
                    A.append(self._normalize(a_close))
                    A.append(self._normalize(a_far))
            return np.stack(A)  # (K, V, V)

    @staticmethod
    def _normalize(A):
        D = np.diag(A.sum(axis=1).clip(min=1) ** -0.5)
        return D @ A @ D

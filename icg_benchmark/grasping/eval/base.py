from typing import TypedDict


class RunStatistics(TypedDict):
    """Struct to store grasp evaluation statistics.

    Attributes:
        sucesses: Number of successful grasps
        tries: Number of attempted grasps
        total_objects: Total number of objects that were spawned
        skips: Number of times a scene was skipped
        num_imgs: Number of images that were rendered
        gripper_collisions: Number of gripper collisions with objects
        object_object_collisions: Number of object-object collisions
        all_collisions: Number of all collisions (gripper + object-object)
    """

    sucesses: int
    tries: int
    total_objects: int
    skips: int
    num_imgs: int
    gripper_collisions: int
    object_object_collisions: int
    all_collisions: int

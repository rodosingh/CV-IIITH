# Video Object Segmentation

Video Segmentation is a task of identifying the set of key objects (with some specific feature properties or semantic value) in a video scene.

Formally, Video Object Segmentation (VOS) extracts generic foreground objects from video sequences with no concern for semantic category recognition. It plays a critical role in a broad range of practical applications, from enhancing visual effects in movie, to understanding scenes in autonomous driving, to virtual background creation in video conferencing, just to name a few.
[PICTURE or GIF]

Based on how much human intervention is involved in inference, VOS models can be divided into three classes (§2.1.2): automatic (AVOS or UVOS, §3.1.1), semi-automatic (SVOS, §3.1.2), and interactive (IVOS, §3.1.3). Moreover, although language-guided video object segmentation (LVOS) falls in the broader category of SVOS, LVOS methods are reviewed alone (§3.1.4), due to the specific multi-modal task setup.

## Our Approach:

We tried to implement two AVOS and one SVOS deep-learning based models along with some simple baselines that leverage upon the optical flow information to segment moving foreground objects.
To know more click read the README.md file located in each folder.
- [Baselines]()
- [RTNet]()
- [MATNet]()
- [TransVOS]()

(Within each folder you will find the individual projects and their README)


from nltk.corpus import wordnet as wn
import nltk
import pandas as pd
import os
from config import *
from robustness.tools.imagenet_helpers import ImageNetHierarchy

nltk.download('wordnet')

in_hier = ImageNetHierarchy("/run/media/riccardo/ea24b431-b1e5-4ec3-95b0-fcbaf83641fb/ImageNet/",
                            '/run/media/riccardo/ea24b431-b1e5-4ec3-95b0-fcbaf83641fb/ImageNet/info')


wn_names = ['animal', 'device', 'conveyance', 'implement', 'container',
            'equipment', 'clothing', 'appliance', 'covering', 'food', 'structure']
wn_ids = ['n00015388', 'n03183080', 'n03100490', 'n03563967', 'n03094503', 'n03294048', 'n03051540', 'n02729837',
          'n03122748', 'n00021265', 'n04341686', 'n04447443', 'n00019128']  # 'n00007846','n09287968','n03405265','n07707451']
# print(superclass_wnid)
# print(label_map)

print(f"Number of root classes: {len(wn_ids)}")

sums = 0
descendants = {}
for ancestor_id in wn_ids:
    for cnt, wnid in enumerate(in_hier.tree[ancestor_id].descendants_all):
        if wnid in in_hier.in_wnids:
            descs = descendants.get(ancestor_id, [])
            descs.append(wnid)
            descendants[ancestor_id] = descs
            sums += 1

print(sums)

classes = os.listdir(
    '/run/media/riccardo/ea24b431-b1e5-4ec3-95b0-fcbaf83641fb/ImageNet/train')

# descnds = []
# for k,v in descendants.items():
#     descnds += v

# print(len(descnds))

# print(set(classes) - set(descnds))

NUM_INTERMEDIATE_CLASSES = 130

hierarchy = []
print(in_hier.in_wnids)
for anchestor, descendant in descendants.items():
    lv1_syn = wn.synset_from_pos_and_offset('n', int(anchestor[1:])).name()
    n_superclasses = round(len(descendant)/sums * NUM_INTERMEDIATE_CLASSES)
    superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(
        n_superclasses=n_superclasses, ancestor_wnid=anchestor, balanced=False)
    for lv2_wnid in superclass_wnid:
        lv2_syn = wn.synset_from_pos_and_offset('n', int(lv2_wnid[1:])).name()
        for class_name in classes:
            if in_hier.is_ancestor(lv2_wnid, class_name):
                fine_syn = wn.synset_from_pos_and_offset(
                    'n', int(class_name[1:])).name()
                hierarchy.append([lv1_syn, lv2_syn, fine_syn])


hierarchy = pd.DataFrame(hierarchy, columns=['lv1', 'lv2', 'fine'])
hierarchy.to_csv('hierarchy.csv', index=False, header=False)

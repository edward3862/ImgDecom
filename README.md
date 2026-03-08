# Cycle-Consistent Tuning for Layered Image Decomposition (CVPR26)

<div align="center">

### [Paper](https://arxiv.org/abs/2602.20989) | [Project Page](https://edward3862.github.io/ImgDecom/)

[Zheng Gu](https://edward3862.github.io/)<sup>1</sup> · [Min Lu](https://deardeer.github.io/)<sup>1</sup> · [Zhida Sun](https://zhdsun.github.io/)<sup>1</sup> · [Dani Lischinski](https://www.cs.huji.ac.il/~danix/)<sup>2</sup> · [Daniel Cohen-Or](https://danielcohenor.com/)<sup>3</sup> · [Hui Huang](https://vcc.tech/~huihuang)<sup>1</sup>

<sup>1</sup>Shenzhen University · <sup>2</sup>Hebrew University of Jerusalem · <sup>3</sup>Tel-Aviv University

</div>

---

<div align="center">
  <img src="./static/images/teaser.jpg" width="100%">
</div>

---

## Abstract

We present an **in-context image decomposition framework** that leverages large diffusion foundation models for layered separation. Our method focuses on the challenging case of **logo-object decomposition**, where the goal is to disentangle a logo from the surface on which it appears while faithfully preserving both layers.

Our approach introduces:

- **Cycle-Consistent Tuning**: Joint training of decomposition and composition models with reconstruction consistency
- **LoRA Adaptation**: Lightweight fine-tuning of pretrained diffusion models
- **Progressive Self-Improvement**: Iterative training set augmentation with high-quality model-generated examples

The method achieves accurate and coherent decompositions and generalizes effectively across other decomposition types (intrinsic decomposition, foreground-background separation), suggesting its potential as a unified framework for layered image decomposition.

---

## 🚀 Code & Models

**Code and pretrained models will be released soon.**

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{gu2026cycle,
  title={Cycle-Consistent Tuning for Layered Image Decomposition},
  author={Gu, Zheng and Lu, Min and Sun, Zhida and Lischinski, Dani and Cohen-Or, Daniel and Huang, Hui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

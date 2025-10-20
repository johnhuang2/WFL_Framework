# Vehicular Platooning with Wireless Federated Learning

Implementation of the paper "Enhancing Vehicular Platooning with Wireless Federated Learning: A Resource-Aware Control Framework" published in IEEE/ACM Transactions on Networking.

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 32GB VRAM (recommended: RTX 4090 or higher)
- **RAM**: 32GB or more
- **Storage**: 50GB free space

The system trains 20 concurrent DeepLabV3+ models with ResNet101 backbones, which requires significant GPU memory.

### Software Requirements
- Python 3.10+
- PyTorch 2.6
- torchvision
- scipy
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the **AI4MARS dataset** for semantic segmentation of Martian terrain. The dataset contains:
- **Image Resolution**: 513×513 pixels
- **Classes**: 4 terrain types (Soil, Bedrock, Sand, Big Rock)
- **Training Samples**: ~10,000 labeled images
- **Distribution**: Non-IID distribution across 20 vehicles using Dirichlet allocation (α=0.5)

**Note**: The current data loader interface returns empty data structures for demonstration purposes. To use actual AI4MARS data:
1. Download the dataset from the official NASA/JPL source
2. Place images in `./data/ai4mars/images/` directory
3. Place labels in `./data/ai4mars/labels/` directory
4. Modify `utils/data_loader.py` to load actual data files

## Training

Run the main training script:

```bash
python main.py
```

The training will run for 500 communication rounds with federated learning across 20 vehicles.

## Future Work

We plan to further optimize the codebase in future releases:
- Integration with more efficient federated learning frameworks (e.g., Flower, FedML)
- Distributed training support for multi-GPU setups
- Enhanced gradient compression algorithms
- Real-time visualization dashboard
- Comprehensive benchmarking suite

## Contact

For questions or issues, please contact: **wbeining.ac@gmail.com**

## Citation

If you use this code, please cite:

```bibtex
@article{Wu2025ToN,
  author={Beining Wu and Jun Huang and Qiang Duan and Liang Dong and Zhipeng Cai},
  title={Enhancing Vehicular Platooning with Wireless Federated Learning: A Resource-Aware Control Framework},
  journal={IEEE/ACM Transactions on Networking},
  year={2025},
  note={in print}
}
```

## License

This implementation is provided for research purposes.


## Time log

| Date | Time | Topic | Description | Learned |
|-----|------|------|--------|--------|
| 4.10. | 1 h | tests | write script to ensure that extracted landmarks are correct ||
|| 2 h | model training | initial training algorithm (no optimizer yet) and plotting function + pretraining (model not yet saved) ||
| 6.10. | 3 h | model architecture + image pipeline | create functions to save model parameters (+ tests) and integrate model prediction into detection ||
| 8.10. | 0.5 h | training | flip dataset images and pretrain model with original and flipped images for left hand support ||
|| 3 h | Docker + compatibility | setup Docker for project and add instructions for it in README ||
|| 2.5 h | Adam optimizer | research Adam optimizer create own implementation for backpropagation. at first the network didn't learn with it, but this was resolved by lowering the learning rate ||
|| 2 h | Pylint and coverage | refactor code how pylint sees fit and ensure coverage ||
| 9.10. | 1 h | Web interface | generate web interface for image pipeline with *GitHub Copilot* and modify Python script to include proper documentation and work similar to original pipeline ||
||||||

- **Total:** 15 h

## Report

- **Current todo:** document results with own data, finish testing for training algorithm
- **Progress:** ensure that extracted landmarks are correct, finish backwards pass (Adam optimizer), and pretrain with different options
- **Uncertanties ja difficulties:**
- **Next:** finish testing for training algorithm, try different model architectures to see if they learn to recognize letters better, and finalize of documentation (results, implementation, and testing)

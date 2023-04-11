# Word2Vec (Word Embedding) with TensorFlow 2.0

This repository contains an implementation of the Word2Vec algorithm using TensorFlow 2.0 to compute vector representations of words. The Word2Vec model used is the Skip-Gram model, which is trained on a small chunk of Wikipedia articles (the text8 dataset).

## Background

Word2Vec is a popular word embedding technique that represents words as vectors in a high-dimensional space. These embeddings can be used in various natural language processing tasks, such as sentiment analysis, document classification, and machine translation. The main idea behind Word2Vec is that words with similar meanings tend to occur in similar contexts.

For more information on Word2Vec, please refer to the following research paper:
[Mikolov, Tomas et al. "Efficient Estimation of Word Representations in Vector Space.", 2013](https://arxiv.org/pdf/1301.3781.pdf)

## Getting Started

To run the Word2Vec implementation, simply clone this repository and execute the `word2vec.py` script using Python 3.

### Prerequisites

- Python 3
- TensorFlow 2.0
- NumPy
- urllib
- zipfile

## Implementation Details

- The text8 dataset of Wikipedia articles is downloaded and processed to create a vocabulary of words.
- Rare words with occurrences below the specified threshold are removed.
- A Skip-Gram model is trained on the dataset for a specified number of steps using Stochastic Gradient Descent (SGD) optimization and Noise Contrastive Estimation (NCE) loss.
- The model's performance is evaluated periodically by finding the nearest neighbors of a set of test words based on their vector representations.

## Author

- Aymeric Damien - [GitHub](https://github.com/aymericdamien)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

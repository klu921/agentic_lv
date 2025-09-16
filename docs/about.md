---
layout: page
title: About
permalink: /about/
---

# About This Research

This blog presents our research on long-context video understanding using agentic pipelines that combine Large Language Models (LLMs) and Visual Language Models (VLMs).

## The Problem

Current machine learning models struggle with understanding long videos (multi-hour content) due to:
- Limited context windows in LLMs
- Computational constraints
- Difficulties with temporal recall and reasoning

## Our Solution

We propose an agentic pipeline that:
1. Generates multi-granularity video representations
2. Uses ReACT framework for temporal reasoning
3. Implements a novel critic module for improved accuracy

## Key Innovation

Our critic module introduces a "Reason, Act, Critique, React" cycle that:
- Evaluates agent outputs
- Identifies discrepancies
- Prompts re-evaluation when needed
- Achieves 5% accuracy improvement

## Results

- **65.18%** accuracy on LV-Bench dataset
- Uses only open-source models
- Efficient processing without end-to-end training

## Contact

For questions or collaborations, please contact us at [email@domain.com](mailto:email@domain.com).
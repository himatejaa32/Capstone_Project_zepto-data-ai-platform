# Zepto Data & AI Platform

## Overview

This project implements an end-to-end Data and AI platform for Zepto.

The project is divided into three modules:

1. Data Pipeline
2. Analytics and Machine Learning
3. Support Assistant using RAG and LangGraph

The solution demonstrates web scraping, data cleaning, SQLite database design, SQL analytics, exploratory data analysis, machine learning, embeddings, vector search, retrieval-augmented generation, LangGraph orchestration, and FastAPI deployment.

---

# Project Architecture

```text
                    Zepto Data & AI Platform
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
      Data Pipeline       Analytics       Support Assistant
             |                |                |
             v                v                v
       Books Website      Titanic Data     Policy Documents
             |                |                |
             v                v                v
          SQLite       ML Models + EDA     Embeddings
                                              |
                                              v
                                           ChromaDB
                                              |
                                              v
                                          Retrieval
                                              |
                                              v
                                           LangGraph
                                              |
                                              v
                                           FastAPI



## Submission Status

This repository contains all three required modules:
- `/data_pipeline`
- `/analytics`
- `/support_assistant`

The project was developed using a feature-branch workflow and merged into `main`.                                       
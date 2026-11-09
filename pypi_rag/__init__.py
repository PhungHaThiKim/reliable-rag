"""pypi_rag: one harness to compare LLM-only, Simple RAG, Hybrid RAG, Behavioral KG-RAG and Verified
KG-RAG for malicious PyPI package detection, with ONE prompt, ONE LLM config and ONE metric suite.

    data.load_spec(spec)                    -> [Sample] read statically (never installed / executed)
    analysis.analyze_package(sample)        -> PackageFacts: call sites, behaviors, data flows (AST)
    analysis.package_representation(...)   -> shared package block (C* code slices) for EVERY method
    knowledge.build_kb(...)                 -> text KB (train split + builtin + PyGuard / YARA / advisories)
    graph.build_behavior_graph(...)         -> provenance-carrying behavior graph; graph_evidence -> G* paths
    pipelines.predict_many(method, ...)     -> [PredictionResult]  (e0 e1 e2 e3 e3a e4 + ablations)
    verification.check_claims / apply_policy-> claim labels, abstention (E4)
    evaluation.summarize_method / comparisons -> metrics, paired bootstrap, McNemar

CLI: python -m pypi_rag.cli --help
"""
__version__ = "0.1.0"

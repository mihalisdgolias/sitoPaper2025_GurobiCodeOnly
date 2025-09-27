## Introduction

[cite_start]The **maritime industry** faces unprecedented regulatory pressure to adopt environmentally sustainable practices, fundamentally altering traditional operational frameworks and cooperation dynamics among container terminals[cite: 38, 39]. [cite_start]Recent environmental regulations have created a **two-tier system** where vessels with **Clean Index (CI)** compliance require specialized terminal facilities, equipment, and services that not all terminals can provide[cite: 40]. [cite_start]This regulatory evolution necessitates a comprehensive reexamination of terminal cooperation models, particularly those based on realistic **vessel transfer mechanisms** rather than abstract volume-sharing approaches[cite: 41].

[cite_start]Container terminals, as critical nodes in global supply chains, must balance **operational efficiency** with **environmental compliance** while maintaining competitive cooperation strategies[cite: 42]. [cite_start]The **vessel-based approach** to terminal cooperation, which focuses on actual complete vessel transfers rather than disaggregated volume units, provides the most realistic framework for analyzing these environmental considerations[cite: 43, 44]. [cite_start]Unlike traditional volume-based models, vessel-based formulations directly incorporate vessel-specific requirements, including environmental compliance needs and CI capabilities[cite: 44].

[cite_start]**Cold Ironing (CI)**—also known as “Alternative Maritime Power” (AMP), “Shore-to-Ship Power” (SSP), or “Onshore Power Supply” (OPS)—has emerged as an effective solution for mitigating environmental pollution in ports[cite: 45]. [cite_start]Ships traditionally keep auxiliary engines running while docked, resulting in significant emissions of sulfur oxides ($\text{SO}_{\text{x}}$), nitrogen oxides ($\text{NO}_{\text{x}}$), carbon dioxide ($\text{CO}_{2}$), and particulate matter[cite: 46]. [cite_start]CI offers an alternative by allowing ships to connect to an onshore electrical grid, enabling them to power these functions with electricity produced from cleaner energy sources[cite: 47, 48]. [cite_start]As regulations on maritime emissions tighten, particularly with initiatives from the International Maritime Organization (IMO) to reduce global shipping’s carbon footprint, CI has become a central strategic consideration[cite: 49].

### Research Focus and Model Enhancement (Gurobi MINLP)

[cite_start]This research builds upon the foundational **vessel-based cooperation framework** established by Pujats, et al. [cite: 50][cite_start], extending their work by integrating **CI infrastructure requirements** and **environmental subsidy policies** into the cooperation model[cite: 51]. [cite_start]Our extension maintains this vessel-based foundation while rigorously addressing the critical environmental dimensions that increasingly define modern port operations[cite: 53].

The resulting model is formulated as a comprehensive **Mixed-Integer Nonlinear Programming (MINLP)** problem. We utilize the **Gurobi solver's Piecewise Linear (PWL) capabilities** to ensure an **exact representation** of the non-linear, utilization-dependent cost structure, which reflects economies and diseconomies of scale.

### Strategic Challenges and Contributions

[cite_start]The integration of environmental considerations creates new strategic opportunities and challenges for terminal cooperation[cite: 54]. [cite_start]**CI-Capable (CIC) terminals** can command premium rates for handling environmentally compliant vessels but require substantial infrastructure investments[cite: 55]. [cite_start]**Environmental subsidy policies** represent government interventions designed to accelerate green technology adoption, potentially altering traditional cooperation incentives and profit distributions[cite: 56]. [cite_start]Understanding these dynamics is crucial for terminal operators making investment decisions and port authorities designing effective environmental policies[cite: 57].

[cite_start]Our research addresses fundamental questions about how environmental requirements reshape terminal cooperation[cite: 58]:
1.  [cite_start]**Efficiency vs. Equity:** How do two primary objectives—**total profit maximization ($\text{MAXPROF}$)** and **minimum profit maximization ($\text{MAXMINP}$)**—compare in terms of system efficiency gains versus distributional profit equity under environmental policies[cite: 33, 131]?
2.  [cite_start]**Investment Value:** How do CI infrastructure investments affect vessel transfer patterns, cooperation participation rates (which are 15-25% higher for CIC terminals), and profit premiums (8-18% higher) across terminals with different baseline **productivity levels** (underproductive, productive, overproductive)[cite: 124, 125, 126, 35]?
3.  [cite_start]**Policy Effectiveness:** What role do environmental subsidies play in incentivizing sustainable cooperation, and what are the cost-effectiveness trade-offs between different objectives[cite: 59]?

[cite_start]By solving this comprehensive MINLP model, our findings provide crucial insights for port authorities designing green incentive policies and terminal operators making strategic environmental infrastructure investments in the evolving maritime regulatory landscape[cite: 36].

***

***


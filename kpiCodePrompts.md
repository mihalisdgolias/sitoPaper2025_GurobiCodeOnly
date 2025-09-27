# Terminal Cooperation Analysis KPIs - Feasible Analysis Prompts

## Overview
This document contains prompts for generating Python code to analyze terminal cooperation simulation results, **limited to KPIs that are actually calculable** from the experimental design and output data. The analysis leverages the full factorial experimental structure testing all combinations of subsidies (0, 50, 100), CI configurations (2^n binary combinations), and pricing mechanisms.

---

## 1. CI Configuration Impact Analysis

### Prompt 1.1: CI Density Response Curves
```
Create Python code to analyze how CI adoption density affects cooperation outcomes. Calculate "CI Density" as the percentage of terminals with CI capability (0% to 100%) and analyze:
- System efficiency gain vs CI density curves
- Individual terminal profit changes by CI density
- Transfer fee levels vs CI density
- Gini coefficient vs CI density

Generate response curves showing diminishing/increasing returns to CI adoption.
```

### Prompt 1.2: Optimal CI Configuration Identification
```
Develop code to identify which specific CI configurations work best for different objectives. For each terminal type composition:
- Rank all CI combinations by total system profit (MAXPROF)
- Rank all CI combinations by minimum terminal profit (MAXMIN)
- Identify CI configurations that maximize fairness (minimize Gini)
- Calculate marginal value of adding CI to each terminal position

Create decision matrices showing optimal CI deployment strategies.
```

### Prompt 1.3: CI Capability Utilization Analysis
```
Write Python code to analyze CI subsidy utilization patterns:
- Total CI subsidy revenue per terminal type
- CI subsidy as percentage of total terminal revenue
- Correlation between CI capability and profit gains
- CI subsidy efficiency (subsidy received vs profit improvement)

Calculate which terminal types benefit most from CI subsidies.
```

---

## 2. Subsidy Policy Effectiveness

### Prompt 2.1: Subsidy Response Analysis
```
Create code to analyze subsidy effectiveness across the three tested levels (0, 50, 100):
- System efficiency gains per dollar of subsidy spent
- Participation rates at each subsidy level
- Diminishing returns analysis for subsidy increases
- Break-even subsidy levels for individual terminals

Generate subsidy response curves and optimal subsidy recommendations.
```

### Prompt 2.2: Subsidy Distribution Impact
```
Develop Python functions to analyze how subsidies flow through the system:
- Total subsidy payments by terminal type
- Net subsidy recipients vs payers by terminal characteristics
- Subsidy redistribution effects on profit equality
- Cross-subsidization patterns between terminal types

Calculate subsidy flow matrices and redistribution metrics.
```

### Prompt 2.3: Marginal Subsidy Impact
```
Write code to calculate marginal effects of subsidy increases from 0→50 and 50→100:
- Marginal system efficiency per subsidy dollar
- Marginal fairness improvement (Gini reduction)
- Marginal participation increase
- Cost-effectiveness ratios for different subsidy levels

Generate marginal impact analysis for policy optimization.
```

---

## 3. Terminal Type and Market Structure Analysis

### Prompt 3.1: Terminal Type Performance Patterns
```
Create code to analyze systematic differences between terminal types:
- Average profit changes by terminal type across all scenarios
- Volume redistribution patterns by terminal type
- Transfer fee payment/receipt patterns by type
- Capacity utilization changes by terminal type

Generate terminal type performance profiles and identify winners/losers.
```

### Prompt 3.2: Market Concentration Effects
```
Develop Python functions to analyze market structure impacts using terminal capacity data:
- Calculate Herfindahl-Hirschman Index (HHI) for each scenario
- Correlation between market concentration and cooperation success
- Large vs small terminal benefit patterns
- Impact of terminal size distribution on profit distribution

Create market structure analysis with concentration metrics.
```

### Prompt 3.3: Terminal Size Class Analysis
```
Write code to classify terminals by capacity size and analyze patterns:
- Classify terminals into size quartiles within each scenario
- Profit change patterns by size class
- Volume gains/losses by terminal size
- Transfer fee burden distribution by size

Generate size-stratified cooperation impact analysis.
```

---

## 4. Pricing Mechanism Effectiveness

### Prompt 4.1: Pricing Mechanism Comparison
```
Create comprehensive code to compare the three pricing mechanisms:
- Total system profits achieved by each mechanism
- Fairness outcomes (Gini coefficients) by mechanism
- Transfer fee levels and volatility by mechanism
- Participation rates and stability by mechanism

Generate mechanism performance scorecards across all scenarios.
```

### Prompt 4.2: Transfer Fee Pattern Analysis
```
Develop Python functions to analyze transfer fee structures:
- Average transfer fees by terminal type and pricing mechanism
- Fee payment asymmetries (who pays vs receives)
- Fee levels relative to terminal profits and volumes
- Cross-subsidization patterns revealed by transfer fees

Create transfer fee flow analysis and pattern identification.
```

### Prompt 4.3: Mechanism Robustness Analysis
```
Write code to assess pricing mechanism robustness across different conditions:
- Mechanism performance across different CI densities
- Mechanism performance across different subsidy levels
- Mechanism sensitivity to terminal type compositions
- Consistency of mechanism outcomes across instances

Calculate robustness scores for each pricing mechanism.
```

---

## 5. Operational Efficiency and Volume Analysis

### Prompt 5.1: Volume Redistribution Analysis
```
Create code to track vessel volume movements in cooperation:
- Volume flow matrices (from original assignment to final assignment)
- Net volume gainers vs losers by terminal type
- System-wide volume concentration vs dispersion
- Volume redistribution efficiency metrics

Generate network flow visualizations and redistribution impact analysis.
```

### Prompt 5.2: Capacity Utilization Optimization
```
Develop Python functions to analyze capacity utilization improvements:
- Utilization rate changes by terminal type
- Under-utilized vs over-utilized terminal identification
- System-wide capacity utilization efficiency
- Load balancing effectiveness across scenarios

Calculate utilization optimization metrics and identify bottlenecks.
```

### Prompt 5.3: Operational Performance Metrics
```
Write code to calculate operational KPIs:
- Coefficient of variation in terminal utilizations
- Volume distribution entropy measures
- Capacity slack analysis by terminal type
- System throughput efficiency improvements

Generate operational performance dashboards and optimization assessments.
```

---

## 6. Fairness and Equity Analysis

### Prompt 6.1: Comprehensive Equity Metrics
```
Create code to calculate multiple fairness measures using profit distribution data:
- Gini coefficient analysis across scenarios
- Coefficient of variation for profit distribution
- 90/10 percentile ratios for profit inequality
- Atkinson Index with different inequality aversion parameters

Generate equity scorecards and fairness trend analysis.
```

### Prompt 6.2: Equity-Efficiency Trade-off Analysis
```
Develop Python functions to analyze trade-offs between fairness and efficiency:
- Plot efficiency gains vs Gini coefficient improvements
- Identify Pareto frontier for equity-efficiency combinations
- Calculate trade-off rates (efficiency lost per fairness gained)
- Compare MAXPROF vs MAXMIN objective outcomes

Create equity-efficiency frontier analysis and optimization recommendations.
```

### Prompt 6.3: Terminal Vulnerability Assessment
```
Write code to identify terminals at risk in cooperation arrangements:
- Terminals with consistent profit losses across scenarios
- Terminals with high profit volatility
- Terminals dependent on subsidies for positive outcomes
- Exit risk assessment based on profit stability

Generate vulnerability heat maps and risk profiles for terminals.
```

---

## 7. Scenario Robustness and Cross-Instance Analysis

### Prompt 7.1: Cross-Instance Consistency Analysis
```
Create code to analyze consistency of results across different data instances:
- Coefficient of variation for key metrics across instances
- Identification of robust strategies (consistent across instances)
- Instance-specific vs generalizable patterns
- Statistical significance testing for treatment effects

Generate robustness assessments and confidence intervals for key findings.
```

### Prompt 7.2: Parameter Sensitivity Analysis
```
Develop Python functions for sensitivity analysis using the experimental design:
- Response surface analysis for subsidy × CI density interactions
- Sensitivity of outcomes to terminal type composition
- Interaction effects between pricing mechanisms and subsidies
- Threshold identification for cooperation viability

Calculate parameter sensitivity rankings and interaction effect magnitudes.
```

### Prompt 7.3: Success Factor Identification
```
Write code to identify critical success factors from the experimental data:
- Correlation analysis between scenario parameters and success metrics
- Classification of scenarios into success/failure categories
- Logistic regression for cooperation viability prediction
- Threshold analysis for minimum conditions

Generate predictive models and success probability assessments.
```

---

## 8. Policy and Implementation Analysis

### Prompt 8.1: Policy Recommendation Framework
```
Create code to generate policy recommendations based on experimental results:
- Optimal subsidy levels for different market structures
- Recommended CI deployment sequences
- Pricing mechanism selection criteria
- Policy effectiveness rankings

Generate policy decision support matrices and implementation guidelines.
```

### Prompt 8.2: Scenario Classification and Benchmarking
```
Develop Python functions to classify and benchmark scenarios:
- Best-practice scenario identification
- Performance percentile rankings
- Gap analysis between current and optimal outcomes
- Target setting for policy objectives

Create benchmark reports and performance standards.
```

### Prompt 8.3: Treatment Effect Analysis
```
Write code to isolate treatment effects from the experimental design:
- Pure subsidy effects (holding CI and pricing constant)
- Pure CI effects (holding subsidy and pricing constant)
- Pure pricing mechanism effects (holding other factors constant)
- Interaction effects between treatments

Calculate treatment effect sizes and statistical significance.
```

---

## 9. Value Creation and Attribution Analysis

### Prompt 9.1: Value Decomposition Analysis
```
Create code to decompose total value creation into components:
- Value from optimal vessel reassignment
- Value from CI technology utilization
- Value from subsidy transfers
- Value from coordination/cooperation itself

Generate value attribution analysis showing sources of improvement.
```

### Prompt 9.2: Terminal-Level Impact Analysis
```
Develop Python functions for detailed terminal-level analysis:
- Profit change decomposition by source (operations vs transfers)
- Volume change impact on terminal profitability
- Transfer fee impact on terminal economics
- Net benefit calculation including all effects

Calculate comprehensive terminal impact assessments.
```

### Prompt 9.3: System-Level Performance Metrics
```
Write code to calculate aggregate system performance:
- Total system profit improvements
- Aggregate capacity utilization improvements
- System-wide efficiency gains
- Social welfare improvements (accounting for subsidies)

Generate system-level performance dashboards and trend analysis.
```

---

## Usage Instructions

### Input Data Requirements:
- Processed DataFrame from existing analysis code with fields:
  - Terminal profits (before/after), volumes, transfer fees
  - Terminal types, CI capabilities, subsidy levels
  - Pricing mechanisms, feasibility status
  - Instance identifiers and scenario parameters

### Output Specifications:
- **Tables**: Excel/CSV format with clear headers and statistical summaries
- **Visualizations**: PNG charts with proper legends showing key relationships
- **Reports**: Summary statistics and findings in structured format
- **Dashboards**: Combined metrics for policy maker consumption

### Code Requirements:
- Integration with existing analysis framework
- Error handling for missing/invalid data
- Statistical testing where appropriate
- Modular design for easy extension and modification

All analysis should focus on **policy-actionable insights** that can guide terminal cooperation implementation and subsidy policy design based on the experimental evidence.
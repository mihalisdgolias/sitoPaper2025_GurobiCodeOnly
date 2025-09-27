# Two-Terminal Cooperation Example: Illustrating Vessel-Based Terminal Cooperation with Environmental Integration

## Overview

This example demonstrates the core concepts of vessel-based terminal cooperation using two container terminals with different operational characteristics. The scenario illustrates how cooperation can improve system efficiency while addressing environmental compliance requirements.

## Terminal Profiles

### Terminal A (Harbor View Terminal)
- **Type**: Underproductive
- **Capacity**: 100,000 TEU/week
- **Initial Utilization**: 40% (40,000 TEU)
- **CI Capability**: Yes (CI-capable)
- **Productivity Profile**: Operates below optimal efficiency due to low demand

### Terminal B (Maritime Gateway Terminal)  
- **Type**: Overproductive
- **Capacity**: 80,000 TEU/week
- **Initial Utilization**: 95% (76,000 TEU)
- **CI Capability**: No
- **Productivity Profile**: Operating near capacity with congestion effects

## Initial State (No Cooperation)

### Vessel Assignments (Baseline)

**Terminal A Vessels:**
- Vessel A1: 15,000 TEU (CI-capable)
- Vessel A2: 12,000 TEU (CI-capable)
- Vessel A3: 13,000 TEU (non-CI)
- **Total Volume**: 40,000 TEU

**Terminal B Vessels:**
- Vessel B1: 18,000 TEU (non-CI)
- Vessel B2: 16,000 TEU (non-CI)
- Vessel B3: 20,000 TEU (CI-capable)
- Vessel B4: 22,000 TEU (non-CI)
- **Total Volume**: 76,000 TEU

### Cost Functions

**Terminal A Cost Structure:**
- Optimal utilization point: 60%
- Currently at 40% (below optimal - economies of scale available)
- Marginal cost at 40%: $180/TEU
- Average cost at 40%: $220/TEU

**Terminal B Cost Structure:**
- Optimal utilization point: 70%
- Currently at 95% (above optimal - congestion effects)
- Marginal cost at 95%: $380/TEU
- Average cost at 95%: $290/TEU

### Initial Profits (No Cooperation)

**Terminal A:**
- Revenue: $400/TEU × 40,000 TEU = $16,000,000
- Cost: $220/TEU × 40,000 TEU = $8,800,000
- **Profit**: $7,200,000

**Terminal B:**
- Revenue: $450/TEU × 76,000 TEU = $34,200,000
- Cost: $290/TEU × 76,000 TEU = $22,040,000
- **Profit**: $12,160,000

**Total System Profit**: $19,360,000

## Cooperation Scenario: Vessel Transfer

### Environmental Policy Context
- CI Subsidy: $50/TEU for CI-capable vessels at CI-capable terminals
- Environmental compliance requirements becoming stricter

### Proposed Vessel Transfer
**Transfer Vessel B3 (20,000 TEU, CI-capable) from Terminal B to Terminal A**

### Post-Cooperation State

**Terminal A (After Receiving Vessel B3):**
- New Volume: 40,000 + 20,000 = 60,000 TEU
- New Utilization: 60% (exactly at optimal point)
- New Marginal Cost: $150/TEU
- New Average Cost: $190/TEU

**Terminal B (After Losing Vessel B3):**
- New Volume: 76,000 - 20,000 = 56,000 TEU  
- New Utilization: 70% (exactly at optimal point)
- New Marginal Cost: $220/TEU
- New Average Cost: $245/TEU

## Economic Analysis of Cooperation

### Transfer Fee Calculation

**Marginal Cost Pricing Mechanism:**
- Terminal B's marginal cost at new utilization (70%): $220/TEU
- Transfer fee: $220/TEU

**Environmental Subsidy:**
- Terminal A receives: $50/TEU × 20,000 TEU = $1,000,000
- (for handling CI-capable vessel B3 with CI infrastructure)

### Profit Changes

**Terminal A (Post-Cooperation):**
- Revenue: $400/TEU × 60,000 TEU = $24,000,000
- Cost: $190/TEU × 60,000 TEU = $11,400,000
- Transfer fee received: $220/TEU × 20,000 TEU = $4,400,000
- Environmental subsidy: $1,000,000
- **New Profit**: $24,000,000 - $11,400,000 + $4,400,000 + $1,000,000 = $18,000,000
- **Profit Increase**: $18,000,000 - $7,200,000 = $10,800,000 (+150%)

**Terminal B (Post-Cooperation):**
- Revenue: $450/TEU × 56,000 TEU = $25,200,000
- Cost: $245/TEU × 56,000 TEU = $13,720,000
- Transfer fee paid: $220/TEU × 20,000 TEU = $4,400,000
- **New Profit**: $25,200,000 - $13,720,000 - $4,400,000 = $7,080,000
- **Profit Change**: $7,080,000 - $12,160,000 = -$5,080,000 (-42%)

### System Performance Comparison

| Metric | No Cooperation | With Cooperation | Change |
|--------|----------------|------------------|---------|
| Terminal A Profit | $7,200,000 | $18,000,000 | +$10,800,000 |
| Terminal B Profit | $12,160,000 | $7,080,000 | -$5,080,000 |
| **Total System Profit** | **$19,360,000** | **$25,080,000** | **+$5,720,000 (+30%)** |
| Terminal A Utilization | 40% | 60% | +20 points |
| Terminal B Utilization | 95% | 70% | -25 points |

## Key Cooperation Insights

### 1. Efficiency Gains
- **System profit increases by 30%** through better capacity utilization
- Both terminals move to their optimal utilization points
- Terminal A eliminates underutilization; Terminal B eliminates congestion

### 2. Environmental Benefits
- CI-capable vessel B3 can now utilize CI services at Terminal A
- Environmental subsidy provides additional incentive for sustainable operations
- System reduces overall emissions through proper CI infrastructure utilization

### 3. Participation Constraint Challenge
Terminal B experiences a significant profit reduction (-42%), violating the 99% participation constraint:
- Required minimum profit: 0.99 × $12,160,000 = $12,038,400
- Actual profit: $7,080,000
- **Constraint violation**: -$4,958,400

### 4. Potential Solutions

**Adjusted Transfer Fee:**
To satisfy participation constraint, transfer fee could be reduced:
- Required Terminal B profit: $12,038,400
- Maximum acceptable transfer fee: ~$50/TEU
- This would reduce Terminal A's gains but maintain cooperation feasibility

**Alternative Objective (MAXMIN):**
MAXMIN objective would optimize for Terminal B's profit, potentially resulting in:
- More balanced profit distribution
- Lower system efficiency but better coalition stability
- Enhanced long-term cooperation sustainability

### 5. Environmental Policy Impact
- CI subsidy of $50/TEU provides $1M additional revenue to Terminal A
- Creates incentive for CI infrastructure investment
- Demonstrates how environmental policies can facilitate cooperation

## Conclusion

This example illustrates the complex trade-offs in terminal cooperation:
- **Efficiency vs. Equity**: Maximum system gains may require redistributive mechanisms
- **Environmental Integration**: CI capabilities create new value streams and cooperation opportunities  
- **Coalition Stability**: Participation constraints are essential for sustainable cooperation
- **Policy Design**: Environmental subsidies can align efficiency and sustainability goals

The vessel-based approach captures these real-world complexities more accurately than volume-based models, providing actionable insights for terminal operators and port authorities designing cooperation frameworks.
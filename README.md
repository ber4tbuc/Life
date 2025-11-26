# 🧬 Life Evolution Simulator

A comprehensive agent-based evolution simulation with reinforcement learning, ecosystem dynamics, and scientific population genetics analysis.

## 🌟 Features

### Core Evolution
- **Genetic Algorithm**: Genome-based evolution with mutation and crossover
- **Phenotype Expression**: Speed, vision, size, and color traits
- **Natural Selection**: Fitness-based survival and reproduction
- **Phylogenetic Tree**: Track evolutionary lineages and generations

### Reinforcement Learning
- **Q-Learning**: Agents learn from experience using Q-tables
- **Memory System**: Short-term and long-term memory for important experiences
- **Learned Behaviors**: Agents adapt their behavior based on past rewards
- **Exploration vs Exploitation**: Epsilon-greedy action selection

### Ecosystem Dynamics
- **Predator-Prey Relationships**: Hunting and evasion mechanics
- **Mutualism/Parasitism**: Symbiotic relationships with energy transfer
- **Trophic Levels**: Food chain with producers, herbivores, and predators
- **Habitat Specialization**: Agents adapt to different environmental conditions
- **Environmental Gradients**: Temperature, pH, and nutrient gradients affect energy costs

### Cultural Evolution
- **Social Learning**: Agents learn cultural traits from nearby agents
- **Teaching/Learning Ability**: Individual variation in cultural transmission
- **Cultural Memory**: Track cultural exchanges and trait propagation

### Scientific Analysis
- **Population Genetics**: Genetic diversity, allele frequencies, Hardy-Weinberg equilibrium
- **Demographics**: Age structure, birth/death rates, life tables
- **Fitness Distribution**: Survival and reproductive fitness tracking
- **Real-time Statistics**: Comprehensive panels for individual agents and global population

### Visualization
- **Interactive World**: Click to add food/poison, Ctrl+Click to inspect agents
- **Phylogenetic Tree**: Visualize evolutionary relationships
- **Statistics Panels**: Detailed metrics for agents and population
- **Color-coded Agents**: Visual representation of genetic traits

## 🚀 Installation

### Requirements
- Python 3.8 or higher
- pygame >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/life-evolution-simulator.git
cd life-evolution-simulator

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python life.py
```

## 🎮 Controls

- **Mouse Left Click**: Add food cluster
- **Mouse Right Click**: Add poison cluster
- **Ctrl + Left Click**: Select agent (view detailed stats)
- **S Key**: Toggle global statistics panel
- **P Key**: Pause/Resume simulation
- **Arrow Keys**: Navigate phylogenetic tree
- **ESC**: Exit simulation

## 📊 Statistics Panels

### Individual Agent Panel (Ctrl+Click)
- Fitness metrics (survival, reproductive, total)
- Genome and phenotype information
- Reinforcement learning metrics (Q-table size, total reward, exploration rate)
- Memory system stats (long-term memory, learned behaviors)
- Cultural evolution data (traits, teaching/learning ability)
- Predator-prey role and success rates
- Symbiotic relationships
- Trophic level and photosynthesis
- Habitat specialization
- Current environmental conditions

### Global Statistics Panel (S Key)
- Population size and demographics
- Average fitness and genetic diversity
- RL metrics across all agents
- Memory and cultural evolution statistics
- Ecosystem dynamics (predator-prey ratios, trophic distribution)
- Environmental gradient information
- Phylogenetic tree visualization

## 🔬 Scientific Features

### Population Genetics
- **Genetic Diversity**: Heterozygosity tracking
- **Allele Frequencies**: Monitor gene frequencies over time
- **Hardy-Weinberg Equilibrium**: Test for evolutionary forces
- **Effective Population Size (Ne)**: Track genetic drift potential
- **Inbreeding Coefficient (F)**: Measure of genetic relatedness

### Evolution Mechanisms
- **Mutation**: Random genome changes
- **Crossover**: Sexual reproduction with gene recombination
- **Selection**: Fitness-based survival and reproduction
- **Genetic Drift**: Random changes in small populations
- **Gene Flow**: Migration between regions

### Ecosystem Modeling
- **Resource Competition**: Limited food availability
- **Energy Budget**: Metabolic costs and energy storage
- **Environmental Adaptation**: Temperature, pH, oxygen tolerance
- **Spatial Structure**: Regional isolation and migration

## 🎯 Use Cases

- **Education**: Learn about evolution, genetics, and ecology
- **Research**: Study population dynamics and evolutionary algorithms
- **Experimentation**: Test different parameters and observe outcomes
- **Visualization**: Watch evolution in real-time

## 📝 Configuration

Key parameters can be adjusted in `life.py`:

- `WORLD_WIDTH`, `WORLD_HEIGHT`: Simulation world size
- `INITIAL_POPULATION`: Starting number of agents
- `mutation_rate`: Probability of genetic mutations
- `sexual_energy_threshold`: Energy required for sexual reproduction
- `asexual_energy_threshold`: Energy required for asexual reproduction
- `reproduction_cooldown_ticks`: Time between reproductions
- `max_population`: Maximum population limit




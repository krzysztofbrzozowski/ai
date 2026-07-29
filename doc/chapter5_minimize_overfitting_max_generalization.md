# Dense(32) vs latent manifold (ukryta rozmaitość, rozmaitość małowymiarowa?)

**Tak, ale precyzyjniej:** `Dense(32)` tworzy **32-wymiarową reprezentację latentną** obrazu, a niekoniecznie samą rozmaitość latentną

```python
layers.Dense(32, activation="relu")
```

przekształca:

```text
784 wartości pikseli → 32 wyuczone wartości
```

Dla jednego obrazu:

```text
x ∈ R⁷⁸⁴
```

warstwa oblicza:

```text
z = ReLU(xW + b)
```

a wynikiem jest:

```text
z ∈ R³²
```

Czyli obraz cyfry zostaje opisany za pomocą 32 liczb, na przykład:

```python
[0.0, 1.7, 0.3, 0.0, 2.1, ..., 0.8]
```

Te liczby mogą kodować kombinacje cech takich jak:

- obecność pętli
- pionowe i poziome kreski
- nachylenie
- położenie kształtu
- grubość linii

Nie musi jednak być tak, że:

```text
neuron 0 = nachylenie
neuron 1 = grubość
neuron 2 = pętla
```

Zazwyczaj jedna cecha jest zakodowana w wielu neuronach, a jeden neuron uczestniczy w kodowaniu wielu cech

## Gdzie jest manifold?

Rozważ wyjścia `Dense(32)` dla wszystkich prawidłowych obrazów MNIST

Teoretycznie mogą one zajmować całą przestrzeń 32-wymiarową, ale w praktyce prawidłowe cyfry prawdopodobnie zajmują tylko pewien uporządkowany fragment tej przestrzeni

```text
przestrzeń latentna: wszystkie możliwe wektory 32D

latent manifold: fragment przestrzeni 32D zajmowany przez sensowne obrazy cyfr
```

Czyli:

```text
Dense(32) → tworzy przestrzeń/reprezentację latentną

obrazy cyfr po przejściu przez Dense(32)
→ mogą układać się na latent manifold w tej przestrzeni
```

Ważne: zwykła warstwa `Dense(32)` w klasyfikatorze nie ma gwarancji, że nauczy się idealnej, gładkiej i interpretowalnej rozmaitości. Uczy się przede wszystkim takiej reprezentacji, która pomaga rozróżnić cyfry

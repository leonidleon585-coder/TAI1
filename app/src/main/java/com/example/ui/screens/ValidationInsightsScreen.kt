package com.example.ui.screens

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.HelpOutline
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.Percent
import androidx.compose.material.icons.filled.Scale
import androidx.compose.material.icons.filled.TrendingDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState

@Composable
fun ValidationInsightsScreen(
    trainerEngine: TrainerEngine,
    state: TrainingState,
    modifier: Modifier = Modifier
) {
    // Take top 10 character weights sorted descending by magnitude
    val sortedWeights = remember(state.characterWeights) {
        state.characterWeights.toList()
            .sortedByDescending { it.second }
            .take(12)
    }

    val maxWeightVal = remember(sortedWeights) {
        if (sortedWeights.isEmpty()) 1f else sortedWeights.maxOf { it.second }.coerceAtLeast(1e-5f)
    }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF030712)) // Dark cyber Slate
            .padding(horizontal = 16.dp),
        contentPadding = PaddingValues(top = 16.dp, bottom = 100.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Neon Header Banner
        item {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(110.dp),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.3f)),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Analytics,
                            contentDescription = "Analytics Icon",
                            tint = Color(0xFF10B981),
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "VALIDATION & INSIGHTS",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Verify what the generative parameters actually learned by querying weight distributions.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )
                }
            }
        }

        // Live Training Metrics Grid (PPL, Loss, Tokens)
        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    ValidationMetricCard(
                        title = "Tokens Processed",
                        value = "${state.totalTokensProcessed}",
                        description = "Cumulative character parameters trained.",
                        icon = Icons.Default.Layers,
                        modifier = Modifier.weight(1f)
                    )

                    ValidationMetricCard(
                        title = "Active Perplexity",
                        value = if (state.currentPerplexity > 0f) String.format("%.2f", state.currentPerplexity) else "NaN",
                        description = "Sequence branch branching factor (lower is better).",
                        icon = Icons.Default.Scale,
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    ValidationMetricCard(
                        title = "Cross-Entropy Loss",
                        value = if (state.currentLoss > 0f) String.format("%.4f", state.currentLoss) else "0.0000",
                        description = "Log probability convergence delta.",
                        icon = Icons.Default.TrendingDown,
                        modifier = Modifier.weight(1f)
                    )

                    ValidationMetricCard(
                        title = "Unique Vocabulary",
                        value = "${state.characterWeights.size} Chars",
                        description = "Total character types modeled.",
                        icon = Icons.Default.LockOpen,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // Character Weights Distribution Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Parameter Weight Magnitude Distribution",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "Visualizes the learned embedding vector magnitude for individual characters. Active letters shift magnitudes as prediction patterns settle.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    if (sortedWeights.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(160.dp)
                                .background(Color(0xFF030712), RoundedCornerShape(8.dp))
                                .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Trace model weights by starting a training cycle first.",
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.Gray,
                                textAlign = TextAlign.Center,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            sortedWeights.forEach { (char, weight) ->
                                val normalizedWeight = (weight / maxWeightVal).coerceIn(0f, 1f)
                                CharacterWeightRow(
                                    char = char,
                                    weightValue = weight,
                                    normalizedValue = normalizedWeight
                                )
                            }
                        }
                    }
                }
            }
        }

        // Informative Explainer Pane
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF111827)),
                border = BorderStroke(1.dp, Color(0xFF374151))
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Icon(
                        imageVector = Icons.Default.HelpOutline,
                        contentDescription = "Info icon",
                        tint = Color(0xFF10B981),
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(
                            text = "A Note on Local Perplexity (PPL)",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Text(
                            text = "Perplexity measures the model's confidence in character forecasting. A perplexity of 5.0 means that when predicting the next letter, the network is as 'perplexed' as if choosing uniformly among 5 possible options. Optimization seeks to lower this number toward 1.0.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFF9CA3AF)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ValidationMetricCard(
    title: String,
    value: String,
    description: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = title, style = MaterialTheme.typography.labelSmall, color = Color.Gray, fontWeight = FontWeight.Bold)
                Icon(imageVector = icon, contentDescription = title, modifier = Modifier.size(16.dp), tint = Color(0xFF10B981))
            }
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                fontFamily = FontFamily.Monospace
            )
            Text(
                text = description,
                style = MaterialTheme.typography.labelSmall,
                color = Color.DarkGray
            )
        }
    }
}

@Composable
fun CharacterWeightRow(
    char: Char,
    weightValue: Float,
    normalizedValue: Float
) {
    val animatedProgress by animateFloatAsState(targetValue = normalizedValue, label = "weight_progress")
    val charLabel = when (char) {
        ' ' -> "[space]"
        '\n' -> "[newline]"
        else -> char.toString()
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = charLabel,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(76.dp)
        )

        Spacer(modifier = Modifier.width(8.dp))

        Box(
            modifier = Modifier
                .weight(1f)
                .height(18.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(Color(0xFF030712))
                .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(3.dp))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(animatedProgress)
                    .background(Color(0xFF10B981))
            )
        }

        Spacer(modifier = Modifier.width(10.dp))

        Text(
            text = String.format("%.4f", weightValue),
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
            color = Color(0xFF10B981),
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(54.dp),
            textAlign = TextAlign.End
        )
    }
}

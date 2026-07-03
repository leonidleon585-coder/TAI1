package com.example.ui.screens

import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cached
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import java.io.File

@Composable
fun SettingsScreen(
    trainerEngine: TrainerEngine,
    state: TrainingState,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    // Local Thread Slider state
    var sliderValue by remember(state.numThreads) { mutableStateOf(state.numThreads.toFloat()) }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF030712)) // Deep space slate
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
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings Icon",
                            tint = Color(0xFF10B981),
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "HARDWARE ENGINE CONFIG",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Customize parallel computing threads and manage local storage files.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )
                }
            }
        }

        // 1. Snapdragon Thread Optimization Slider
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "CPU Thread Core Allocation",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF10B981),
                            fontFamily = FontFamily.Monospace
                        )
                        Icon(imageVector = Icons.Default.Memory, contentDescription = "CPU icon", tint = Color(0xFF10B981), modifier = Modifier.size(18.dp))
                    }

                    Text(
                        text = "Fine-tune the count of parallel asynchronous processes allocating gradients. Snapdragon 8s Gen 4 supports optimal execution at 8 to 12 threads.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Allocated: ${sliderValue.toInt()} Threads",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = FontFamily.Monospace
                        )
                        if (sliderValue.toInt() == 8) {
                            Text(
                                text = "Snapdragon Standard (8 Cores)",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color(0xFF10B981),
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }

                    Slider(
                        value = sliderValue,
                        onValueChange = {
                            sliderValue = it
                            trainerEngine.setThreadCount(it.toInt())
                        },
                        valueRange = 1f..16f,
                        steps = 14,
                        colors = SliderDefaults.colors(
                            activeTrackColor = Color(0xFF10B981),
                            inactiveTrackColor = Color(0xFF1E293B),
                            thumbColor = Color(0xFF10B981)
                        )
                    )
                }
            }
        }

        // 2. Weights Serialization & Storage Export
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Save & Export Weights",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "Export fully optimized fine-tuned neural network parameters (.tflite format) to local storage. This runs completely local and is encrypted.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )

                    Button(
                        onClick = {
                            val file = trainerEngine.exportTrainedModel()
                            if (file != null) {
                                Toast.makeText(context, "Model weights exported to Documents successfully!", Toast.LENGTH_LONG).show()
                            } else {
                                Toast.makeText(context, "Export failed. Please execute training first to shape weights.", Toast.LENGTH_SHORT).show()
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Save, contentDescription = "Export weights icon", modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("EXPORT PARAMETERS TO DOCUMENTS", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                    }

                    if (state.lastExportedPath != null) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Color(0xFF052E16), RoundedCornerShape(8.dp))
                                .border(1.dp, Color(0xFF15803D), RoundedCornerShape(8.dp))
                                .padding(10.dp)
                        ) {
                            Text(
                                text = "Saved offline: \n${state.lastExportedPath}",
                                style = MaterialTheme.typography.labelSmall,
                                fontFamily = FontFamily.Monospace,
                                color = Color(0xFF86EFAC)
                            )
                        }
                    }
                }
            }
        }

        // 3. Clear System Caches
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Local Storage Cache",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "Clears loaded external TXT documents, scraped datasets, and returns parameters to standard base-built values.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )

                    Button(
                        onClick = {
                            trainerEngine.clearCache()
                            Toast.makeText(context, "Engine cache and model buffers flushed.", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF7F1D1D)),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Cached, contentDescription = "Clear cache icon", modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("FLUSH ENGINE CACHES", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold, color = Color.White)
                    }
                }
            }
        }
    }
}

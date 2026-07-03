package com.example.ui.screens

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import com.example.ml.WebScraper
import kotlinx.coroutines.launch

@Composable
fun TrainingHubScreen(
    trainerEngine: TrainerEngine,
    state: TrainingState,
    onPickModel: () -> Unit,
    onPickDataset: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Hyperparameters states
    var learningRateStr by remember { mutableStateOf("0.02") }
    var epochsStr by remember { mutableStateOf("15") }
    var batchSizeStr by remember { mutableStateOf("32") }

    // Web Scraping state
    var websiteUrl by remember { mutableStateOf("") }
    var isScraping by remember { mutableStateOf(false) }
    var scrapedPreviewText by remember { mutableStateOf("") }
    var showPreviewWindow by remember { mutableStateOf(false) }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF030712)) // Deep space slate
            .padding(horizontal = 16.dp),
        contentPadding = PaddingValues(top = 16.dp, bottom = 100.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Neon header branding
        item {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(110.dp),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.3f)), // Neon green glow border
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A))
            ) {
                Box(modifier = Modifier.fillMaxSize()) {
                    // Cyber background grids
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Terminal,
                                contentDescription = "Terminal Icon",
                                tint = Color(0xFF10B981),
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "TRAINING HUB",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Optimize model weights offline using custom text datasets or raw web corpus.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFF9CA3AF)
                        )
                    }
                }
            }
        }

        // Engine Configuration (Pickers)
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Engine Core Selection",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )

                    // Active model display
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF030712), RoundedCornerShape(8.dp))
                            .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (state.isCustomModel) Icons.Default.DataObject else Icons.Default.Memory,
                            contentDescription = "Base Model Type",
                            tint = Color(0xFF10B981),
                            modifier = Modifier.size(22.dp)
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Active Model File",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color(0xFF6B7280)
                            )
                            Text(
                                text = state.baseModelName,
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Medium,
                                color = Color.White
                            )
                        }
                        if (state.isCustomModel) {
                            IconButton(
                                onClick = { trainerEngine.useBuiltInModel() },
                                modifier = Modifier.size(32.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Refresh,
                                    contentDescription = "Reset Model",
                                    tint = Color(0xFFEF4444),
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                        }
                    }

                    // Source Selectors
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = onPickModel,
                            modifier = Modifier
                                .weight(1f)
                                .height(46.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.FileOpen,
                                contentDescription = "Model picker icon",
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Base Model", style = MaterialTheme.typography.bodySmall)
                        }

                        Button(
                            onClick = onPickDataset,
                            modifier = Modifier
                                .weight(1f)
                                .height(46.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.Source,
                                contentDescription = "Dataset picker icon",
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Local TXT", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        // Advanced Web Scraping Ingestion Engine
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Web Scraping Dataset Engine",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "Input multiple website links (separated by commas or newlines) to extract pure, prose training characters, autonomously discover up to 5 child links per domain, and queue download them sequential-style.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF9CA3AF)
                    )

                    val queueIsProcessing = state.activeUrlProcessing != "Idle" && state.activeUrlProcessing != "Completed" && state.activeUrlProcessing != ""

                    OutlinedTextField(
                        value = websiteUrl,
                        onValueChange = { websiteUrl = it },
                        placeholder = { Text("https://example.com/blog\nhttps://example.com/docs", color = Color.Gray, style = MaterialTheme.typography.bodySmall) },
                        singleLine = false,
                        maxLines = 4,
                        enabled = !queueIsProcessing && !state.isTraining,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(100.dp),
                        shape = RoundedCornerShape(8.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFF10B981),
                            unfocusedBorderColor = Color(0xFF334155),
                            focusedLabelColor = Color(0xFF10B981)
                        )
                    )

                    Button(
                        onClick = {
                            if (websiteUrl.trim().isEmpty()) {
                                Toast.makeText(context, "Please enter valid URLs first.", Toast.LENGTH_SHORT).show()
                                return@Button
                            }
                            trainerEngine.processScraperQueue(websiteUrl)
                        },
                        enabled = !queueIsProcessing && !state.isTraining,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        if (queueIsProcessing) {
                            CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.Black, strokeWidth = 2.dp)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("INGESTING QUEUE...", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = Color.Black)
                        } else {
                            Icon(
                                imageVector = Icons.Default.CloudDownload,
                                contentDescription = "Scrape queue content",
                                tint = Color.Black,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("RUN SCRApER QUEUE", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = Color.Black, fontFamily = FontFamily.Monospace)
                        }
                    }
                }
            }
        }

        // Background Queue Status Dashboard Indicators
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Background Queue Status",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MiniDashboardMetric(
                            title = "Tokens Downloaded",
                            value = if (state.totalTokensDownloaded > 0) String.format("%,d", state.totalTokensDownloaded) else "0",
                            icon = Icons.Default.DownloadDone,
                            modifier = Modifier.weight(1f)
                        )

                        MiniDashboardMetric(
                            title = "Links in Queue",
                            value = "${state.discoveredLinksInQueue} URLs",
                            icon = Icons.Default.Link,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF030712), RoundedCornerShape(8.dp))
                            .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
                            .padding(12.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            val isActive = state.activeUrlProcessing != "Idle" && state.activeUrlProcessing != "Completed" && state.activeUrlProcessing != ""
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(if (isActive) Color(0xFF10B981) else Color.Gray)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    text = "Active URL Processing",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.Gray,
                                    fontFamily = FontFamily.Monospace
                                )
                                Text(
                                    text = state.activeUrlProcessing,
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Bold,
                                    color = Color.White,
                                    fontFamily = FontFamily.Monospace,
                                    maxLines = 1
                                )
                            }
                        }
                    }
                }
            }
        }

        // Hyperparameter Tuning Configurations
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Hyperparameter Settings",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF10B981),
                        fontFamily = FontFamily.Monospace
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedTextField(
                            value = learningRateStr,
                            onValueChange = { learningRateStr = it },
                            label = { Text("Learning Rate", style = MaterialTheme.typography.labelSmall) },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(8.dp),
                            enabled = !state.isTraining,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF10B981),
                                unfocusedBorderColor = Color(0xFF334155),
                                focusedLabelColor = Color(0xFF10B981)
                            )
                        )

                        OutlinedTextField(
                            value = epochsStr,
                            onValueChange = { epochsStr = it },
                            label = { Text("Epochs", style = MaterialTheme.typography.labelSmall) },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(8.dp),
                            enabled = !state.isTraining,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF10B981),
                                unfocusedBorderColor = Color(0xFF334155),
                                focusedLabelColor = Color(0xFF10B981)
                            )
                        )

                        OutlinedTextField(
                            value = batchSizeStr,
                            onValueChange = { batchSizeStr = it },
                            label = { Text("Batch Size", style = MaterialTheme.typography.labelSmall) },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(8.dp),
                            enabled = !state.isTraining,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF10B981),
                                unfocusedBorderColor = Color(0xFF334155),
                                focusedLabelColor = Color(0xFF10B981)
                            )
                        )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    if (state.isTraining) {
                        Button(
                            onClick = { trainerEngine.stopTraining() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Icon(imageVector = Icons.Default.Pause, contentDescription = "Pause training")
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("PAUSE BACKPROPAGATION", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color.White)
                        }
                    } else {
                        Button(
                            onClick = {
                                val lr = learningRateStr.toFloatOrNull() ?: 0.02f
                                val ep = epochsStr.toIntOrNull() ?: 15
                                val bs = batchSizeStr.toIntOrNull() ?: 32
                                if (lr <= 0f || ep <= 0 || bs <= 0) {
                                    Toast.makeText(context, "Invalid hyperparameters specified.", Toast.LENGTH_SHORT).show()
                                } else {
                                    trainerEngine.startTraining(ep, bs, lr)
                                }
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Icon(imageVector = Icons.Default.PlayArrow, contentDescription = "Start optimization", tint = Color.Black)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("RUN LOCAL TRAINING", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color.Black)
                        }
                    }
                }
            }
        }

        // Training Real-Time Dashboard
        item {
            com.example.ui.screens.DashboardCard(state = state)
        }

        // On-Device Compilation Terminal
        item {
            com.example.ui.screens.LogsCard(state = state)
        }
    }
}

@Composable
fun DashboardCard(state: TrainingState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Neural Matrix Dashboard",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF10B981),
                fontFamily = FontFamily.Monospace
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                com.example.ui.screens.MiniDashboardMetric(
                    title = "Epoch Range",
                    value = "${state.currentEpoch}/${state.totalEpochs}",
                    icon = Icons.Default.Sync,
                    modifier = Modifier.weight(1f)
                )

                com.example.ui.screens.MiniDashboardMetric(
                    title = "Live Loss",
                    value = if (state.currentLoss > 0f) String.format("%.4f", state.currentLoss) else "0.0000",
                    icon = Icons.Default.TrendingDown,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                com.example.ui.screens.MiniDashboardMetric(
                    title = "Sequence Match",
                    value = if (state.accuracy > 0) String.format("%.2f%%", state.accuracy * 100f) else "0.00%",
                    icon = Icons.Default.CheckCircle,
                    modifier = Modifier.weight(1f)
                )

                com.example.ui.screens.MiniDashboardMetric(
                    title = "Elapsed",
                    value = String.format("%.1fs", state.elapsedMs / 1000f),
                    icon = Icons.Default.Timer,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                com.example.ui.screens.MiniDashboardMetric(
                    title = "Total Params (M)",
                    value = if (state.totalTrainableParameters > 0) String.format("%,d", state.totalTrainableParameters) else "0",
                    icon = Icons.Default.Settings,
                    modifier = Modifier.weight(1f)
                )

                com.example.ui.screens.MiniDashboardMetric(
                    title = "Throughput Speed",
                    value = if (state.throughputSpeed > 0) String.format("%.1f Tok/s", state.throughputSpeed) else "0.0 Tok/s",
                    icon = Icons.Default.Speed,
                    modifier = Modifier.weight(1f)
                )
            }

            if (state.totalEpochs > 0) {
                val progress = state.currentEpoch.toFloat() / state.totalEpochs.toFloat()
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Converging Progress", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                        Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelSmall, color = Color(0xFF10B981))
                    }
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp)),
                        color = Color(0xFF10B981),
                        trackColor = Color(0xFF1E293B)
                    )
                }
            }

            Text(
                text = "Cross-Entropy Convergence History",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
                fontFamily = FontFamily.Monospace
            )
            com.example.ui.screens.EmbeddedLossGraph(lossHistory = state.lossHistory)
        }
    }
}

@Composable
fun MiniDashboardMetric(title: String, value: String, icon: androidx.compose.ui.graphics.vector.ImageVector, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .background(Color(0xFF030712), RoundedCornerShape(8.dp))
            .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
            .padding(12.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = title, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                Icon(imageVector = icon, contentDescription = title, modifier = Modifier.size(14.dp), tint = Color(0xFF10B981))
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
    }
}

@Composable
fun EmbeddedLossGraph(lossHistory: List<Float>, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(110.dp)
            .background(Color(0xFF030712), shape = RoundedCornerShape(8.dp))
            .border(1.dp, Color(0xFF1E293B), shape = RoundedCornerShape(8.dp))
            .padding(10.dp)
    ) {
        if (lossHistory.size < 2) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text = "Awaiting optimization epochs to trace curve...",
                    color = Color(0xFF4B5563),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace
                )
            }
        } else {
            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                val width = size.width
                val height = size.height
                val pointsCount = lossHistory.size

                val maxLoss = lossHistory.maxOrNull() ?: 1.0f
                val minLoss = lossHistory.minOrNull() ?: 0.0f
                val lossRange = if (maxLoss - minLoss == 0f) 1.0f else maxLoss - minLoss

                val path = androidx.compose.ui.graphics.Path()
                for (i in lossHistory.indices) {
                    val x = i * (width / (pointsCount - 1))
                    val normalizedY = (lossHistory[i] - minLoss) / lossRange
                    val y = height - (normalizedY * height)

                    if (i == 0) {
                        path.moveTo(x, y)
                    } else {
                        path.lineTo(x, y)
                    }
                }

                val fillPath = androidx.compose.ui.graphics.Path().apply {
                    addPath(path)
                    lineTo(width, height)
                    lineTo(0f, height)
                    close()
                }
                drawPath(
                    path = fillPath,
                    brush = Brush.verticalGradient(
                        colors = listOf(Color(0xFF10B981).copy(alpha = 0.15f), Color.Transparent)
                    )
                )

                drawPath(
                    path = path,
                    color = Color(0xFF10B981),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(
                        width = 2.dp.toPx(),
                        cap = androidx.compose.ui.graphics.StrokeCap.Round
                    )
                )
            }
        }
    }
}

@Composable
fun LogsCard(state: TrainingState) {
    val listState = androidx.compose.foundation.lazy.rememberLazyListState()

    LaunchedEffect(state.logs.size) {
        if (state.logs.isNotEmpty()) {
            listState.animateScrollToItem(state.logs.size - 1)
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Matrix Compilation Terminal",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF10B981),
                    fontFamily = FontFamily.Monospace
                )
                Row(
                    modifier = Modifier
                        .background(Color(0xFF030712), RoundedCornerShape(12.dp))
                        .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(12.dp))
                        .padding(horizontal = 8.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(if (state.isTraining) Color(0xFF10B981) else Color(0xFFEF4444))
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = if (state.isTraining) "ACTIVE" else "STANDBY",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.LightGray,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(130.dp)
                    .background(Color(0xFF030712), RoundedCornerShape(8.dp))
                    .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
                    .padding(8.dp)
            ) {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(state.logs.size) { index ->
                        val log = state.logs[index]
                        Text(
                            text = "> $log",
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            color = if (log.contains("FATAL") || log.contains("Error")) Color(0xFFEF4444) else Color(0xFFE5E7EB)
                        )
                    }
                    if (state.logs.isEmpty()) {
                        item {
                            Text(
                                text = "> Compilation core idling. Setup dataset or start training...",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                color = Color.Gray
                            )
                        }
                    }
                }
            }
        }
    }
}

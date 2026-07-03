package com.example

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.TrendingDown
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import com.example.ui.theme.MyApplicationTheme
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    private lateinit var trainerEngine: TrainerEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        trainerEngine = TrainerEngine(applicationContext)

        setContent {
            MyApplicationTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    bottomBar = {
                        AttributionFooter()
                    }
                ) { innerPadding ->
                    MainScreen(
                        trainerEngine = trainerEngine,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun MainScreen(trainerEngine: TrainerEngine, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val state by trainerEngine.state.collectAsState()

    // Hyperparameters states
    var learningRateStr by remember { mutableStateOf("0.02") }
    var epochsStr by remember { mutableStateOf("15") }
    var batchSizeStr by remember { mutableStateOf("32") }

    // Playground state
    var playgroundInput by remember { mutableStateOf("deep inside") }
    var playgroundOutput by remember { mutableStateOf("Click Generate to run auto-regressive prediction locally.") }

    // File pickers launchers
    val datasetPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val text = context.contentResolver.openInputStream(it)?.bufferedReader()?.use { reader ->
                    reader.readText()
                } ?: ""
                trainerEngine.loadCustomDataset(it, text)
                Toast.makeText(context, "Dataset loaded successfully!", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to read dataset: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val cacheFile = File(context.cacheDir, "temp_model.tflite")
                context.contentResolver.openInputStream(it)?.use { input ->
                    FileOutputStream(cacheFile).use { output ->
                        input.copyTo(output)
                    }
                }
                trainerEngine.loadCustomTfliteModel(it, cacheFile)
                Toast.makeText(context, "LiteRT Model loaded successfully!", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to map model: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    BoxWithConstraints(modifier = modifier.background(MaterialTheme.colorScheme.background)) {
        val isWide = maxWidth > 600.dp
        
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            contentPadding = PaddingValues(top = 16.dp, bottom = 80.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Hero Banner
            item {
                HeroHeader()
            }

            // Architecture Config Panel
            item {
                ModelConfigCard(
                    state = state,
                    onPickModel = { modelPicker.launch("application/octet-stream") },
                    onPickDataset = { datasetPicker.launch("text/*") },
                    onResetModel = { trainerEngine.useBuiltInModel() }
                )
            }

            // Hyperparameters & Action Panel
            item {
                HyperparametersCard(
                    state = state,
                    learningRateStr = learningRateStr,
                    epochsStr = epochsStr,
                    batchSizeStr = batchSizeStr,
                    onLearningRateChange = { learningRateStr = it },
                    onEpochsChange = { epochsStr = it },
                    onBatchSizeChange = { batchSizeStr = it },
                    onStartTraining = {
                        val lr = learningRateStr.toFloatOrNull() ?: 0.02f
                        val ep = epochsStr.toIntOrNull() ?: 15
                        val bs = batchSizeStr.toIntOrNull() ?: 32
                        
                        if (lr <= 0f || ep <= 0 || bs <= 0) {
                            Toast.makeText(context, "Invalid hyperparameters specified.", Toast.LENGTH_SHORT).show()
                        } else {
                            trainerEngine.startTraining(ep, bs, lr)
                        }
                    },
                    onStopTraining = {
                        trainerEngine.stopTraining()
                    }
                )
            }

            // Real-time Optimization Dashboard
            item {
                TrainingDashboardCard(state = state)
            }

            // Logging & Console Console Pane
            item {
                ConsoleLogsCard(state = state)
            }

            // Interactive Live Playground
            item {
                PlaygroundCard(
                    playgroundInput = playgroundInput,
                    playgroundOutput = playgroundOutput,
                    onInputChange = { playgroundInput = it },
                    onEvaluate = {
                        playgroundOutput = trainerEngine.generateText(playgroundInput, length = 80)
                    }
                )
            }

            // Weights Export
            item {
                ExportWeightsCard(
                    state = state,
                    onExport = {
                        val file = trainerEngine.exportTrainedModel()
                        if (file != null) {
                            Toast.makeText(context, "Model weights exported to Documents!", Toast.LENGTH_LONG).show()
                        } else {
                            Toast.makeText(context, "Export failed. Train the model first.", Toast.LENGTH_SHORT).show()
                        }
                    }
                )
            }
        }
    }
}

@Composable
fun HeroHeader() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(180.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            Image(
                painter = painterResource(id = R.drawable.img_hero_banner),
                contentDescription = "Futuristic Neural Network connected nodes",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            // Overlay gradient for readibility
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(Color.Transparent, Color(0xCC000000)),
                            startY = 50f
                        )
                    )
            )
            Column(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Memory,
                        contentDescription = "Snapdragon icon",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "LiteRT ODT Optimizer",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
                Text(
                    text = "High-performance on-device training & fine-tuning engine.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.LightGray
                )
            }
        }
    }
}

@Composable
fun ModelConfigCard(
    state: TrainingState,
    onPickModel: () -> Unit,
    onPickDataset: () -> Unit,
    onResetModel: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Engine Configuration",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )
            
            // Current model state
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = if (state.isCustomModel) Icons.Default.DataObject else Icons.Default.AutoAwesome,
                    contentDescription = "Model Type Icon",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Active Base Model",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray
                    )
                    Text(
                        text = state.baseModelName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                if (state.isCustomModel) {
                    IconButton(
                        onClick = onResetModel,
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Restore Default Model",
                            tint = Color.Red,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }

            // Actions row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = onPickModel,
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(imageVector = Icons.Outlined.FileOpen, contentDescription = "Pick Model icon")
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Base Model", style = MaterialTheme.typography.bodyMedium)
                }

                Button(
                    onClick = onPickDataset,
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B)),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(imageVector = Icons.Outlined.Source, contentDescription = "Pick Dataset icon")
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Dataset", style = MaterialTheme.typography.bodyMedium)
                }
            }
            Text(
                text = "*Loads TXT/JSONL samples. Custom models must expose training signatures.",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray
            )
        }
    }
}

@Composable
fun HyperparametersCard(
    state: TrainingState,
    learningRateStr: String,
    epochsStr: String,
    batchSizeStr: String,
    onLearningRateChange: (String) -> Unit,
    onEpochsChange: (String) -> Unit,
    onBatchSizeChange: (String) -> Unit,
    onStartTraining: () -> Unit,
    onStopTraining: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Optimization Parameters",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedTextField(
                    value = learningRateStr,
                    onValueChange = onLearningRateChange,
                    label = { Text("Learning Rate", style = MaterialTheme.typography.labelSmall) },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                    enabled = !state.isTraining,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        unfocusedBorderColor = Color(0xFF334155)
                    )
                )

                OutlinedTextField(
                    value = epochsStr,
                    onValueChange = onEpochsChange,
                    label = { Text("Epochs", style = MaterialTheme.typography.labelSmall) },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                    enabled = !state.isTraining,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        unfocusedBorderColor = Color(0xFF334155)
                    )
                )

                OutlinedTextField(
                    value = batchSizeStr,
                    onValueChange = onBatchSizeChange,
                    label = { Text("Batch Size", style = MaterialTheme.typography.labelSmall) },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                    enabled = !state.isTraining,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        unfocusedBorderColor = Color(0xFF334155)
                    )
                )
            }

            if (state.isTraining) {
                Button(
                    onClick = onStopTraining,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(imageVector = Icons.Default.Pause, contentDescription = "Pause training")
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("PAUSE OPTIMIZATION", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color.White)
                }
            } else {
                Button(
                    onClick = onStartTraining,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(imageVector = Icons.Default.PlayArrow, contentDescription = "Start optimization", tint = Color.Black)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("START OPTIMIZATION", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color.Black)
                }
            }
        }
    }
}

@Composable
fun TrainingDashboardCard(state: TrainingState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Training Real-Time Dashboard",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )

            // Performance statistics row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Current Epoch",
                    value = "${state.currentEpoch}/${state.totalEpochs}",
                    icon = Icons.Default.Sync,
                    modifier = Modifier.weight(1f)
                )

                MetricCard(
                    title = "Live Loss",
                    value = if (state.currentLoss > 0f) String.format("%.4f", state.currentLoss) else "0.0000",
                    icon = Icons.AutoMirrored.Filled.TrendingDown,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Optimized Acc",
                    value = if (state.accuracy > 0) String.format("%.2f%%", state.accuracy * 100f) else "0.00%",
                    icon = Icons.Default.CheckCircle,
                    modifier = Modifier.weight(1f)
                )

                MetricCard(
                    title = "Elapsed Time",
                    value = formatElapsed(state.elapsedMs),
                    icon = Icons.Default.Timer,
                    modifier = Modifier.weight(1f)
                )
            }

            // Real-time progress bar
            if (state.totalEpochs > 0) {
                val progress = state.currentEpoch.toFloat() / state.totalEpochs.toFloat()
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Epoch Progress", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                        Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                    }
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp)),
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = Color(0xFF1E293B)
                    )
                }
            }

            // Live-updating Loss Curve
            Text(
                text = "Loss Convergence History",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Gray
            )
            LossGraph(lossHistory = state.lossHistory)
        }
    }
}

@Composable
fun MetricCard(title: String, value: String, imageVector: androidx.compose.ui.graphics.vector.ImageVector? = null, icon: androidx.compose.ui.graphics.vector.ImageVector? = null, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
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
                if (icon != null) {
                    Icon(imageVector = icon, contentDescription = title, modifier = Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
    }
}

@Composable
fun LossGraph(lossHistory: List<Float>, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(140.dp)
            .background(Color(0xFF0F172A), shape = RoundedCornerShape(8.dp))
            .border(1.dp, Color(0xFF1E293B), shape = RoundedCornerShape(8.dp))
            .padding(12.dp)
    ) {
        if (lossHistory.size < 2) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text = "Awaiting optimization epochs...",
                    color = Color(0xFF64748B),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        } else {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val width = size.width
                val height = size.height
                val pointsCount = lossHistory.size

                val maxLoss = lossHistory.maxOrNull() ?: 1.0f
                val minLoss = lossHistory.minOrNull() ?: 0.0f
                val lossRange = if (maxLoss - minLoss == 0f) 1.0f else maxLoss - minLoss

                val path = Path()
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

                // Fill under curve
                val fillPath = Path().apply {
                    addPath(path)
                    lineTo(width, height)
                    lineTo(0f, height)
                    close()
                }
                drawPath(
                    path = fillPath,
                    brush = Brush.verticalGradient(
                        colors = listOf(Color(0xFF10B981).copy(alpha = 0.2f), Color.Transparent)
                    )
                )

                // Line stroke
                drawPath(
                    path = path,
                    color = Color(0xFF10B981),
                    style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round)
                )

                // Last point glow indicator
                val lastX = width
                val lastNormY = (lossHistory.last() - minLoss) / lossRange
                val lastY = height - (lastNormY * height)
                drawCircle(
                    color = Color(0xFF3B82F6),
                    radius = 4.dp.toPx(),
                    center = Offset(lastX, lastY)
                )
            }
        }
    }
}

@Composable
fun ConsoleLogsCard(state: TrainingState) {
    val listState = rememberLazyListState()

    // Auto-scroll console logs as they arrive
    LaunchedEffect(state.logs.size) {
        if (state.logs.isNotEmpty()) {
            listState.animateScrollToItem(state.logs.size - 1)
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "On-Device Compilation Console",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary
                )
                Row(
                    modifier = Modifier
                        .background(Color(0xFF1E293B), RoundedCornerShape(12.dp))
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
                        text = if (state.isTraining) "RUNNING" else "STANDBY",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.LightGray
                    )
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp)
                    .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                    .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
                    .padding(8.dp)
            ) {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(state.logs) { log ->
                        Text(
                            text = log,
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            color = if (log.contains("FATAL") || log.contains("Error")) Color(0xFFEF4444) else Color(0xFFE2E8F0)
                        )
                    }
                    if (state.logs.isEmpty()) {
                        item {
                            Text(
                                text = "> Local model training process awaiting triggers...",
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

@Composable
fun PlaygroundCard(
    playgroundInput: String,
    playgroundOutput: String,
    onInputChange: (String) -> Unit,
    onEvaluate: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Model Local Playground",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Enter a seed phrase to predict and append the next 50-100 characters locally based on its weights.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )

            OutlinedTextField(
                value = playgroundInput,
                onValueChange = onInputChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("E.g., deep inside", color = Color.Gray) },
                shape = RoundedCornerShape(8.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = Color(0xFF334155)
                )
            )

            Button(
                onClick = onEvaluate,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                shape = RoundedCornerShape(8.dp)
            ) {
                Icon(imageVector = Icons.Default.FlashOn, contentDescription = "Generate text icon")
                Spacer(modifier = Modifier.width(6.dp))
                Text("GENERATE TEXT PREDICTION", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                    .border(1.dp, Color(0xFF1E293B), RoundedCornerShape(8.dp))
                    .padding(12.dp)
            ) {
                Text(
                    text = playgroundOutput,
                    style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

@Composable
fun ExportWeightsCard(state: TrainingState, onExport: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, Color(0xFF1E293B))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "Save & Export Weights",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Export fully optimized fine-tuned neural network parameters (.tflite format) to local storage. This runs completely local and is encrypted.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )

            Button(
                onClick = onExport,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.tertiary),
                shape = RoundedCornerShape(8.dp)
            ) {
                Icon(imageVector = Icons.Default.Save, contentDescription = "Export weights icon")
                Spacer(modifier = Modifier.width(6.dp))
                Text("EXPORT MODEL TO LOCAL DOCUMENTS", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
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
                        text = "Successfully saved! Path:\n${state.lastExportedPath}",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF86EFAC)
                    )
                }
            }
        }
    }
}

@Composable
fun AttributionFooter() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF070A15))
            .padding(12.dp)
            .navigationBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "Core Engine: zftoz • Snapdragon 8s Gen 4 Optimized • 100% Offline AI",
            style = MaterialTheme.typography.labelSmall,
            color = Color.Gray,
            textAlign = TextAlign.Center
        )
    }
}

private fun formatElapsed(ms: Long): String {
    if (ms <= 0) return "0.0s"
    val sec = ms / 1000f
    return String.format("%.1fs", sec)
}

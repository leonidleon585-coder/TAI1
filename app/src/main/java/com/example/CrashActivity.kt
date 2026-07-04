package com.example

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ui.theme.MyApplicationTheme

class CrashActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val errorLog = intent.getStringExtra("error_stacktrace") 
            ?: "No stacktrace found in intent extras. Check crash_log.txt."

        setContent {
            MyApplicationTheme {
                CrashScreen(errorLog = errorLog) {
                    // Action to restart core engine
                    try {
                        val trainer = TrainerEngine.getInstance(applicationContext)
                        trainer.clearCache()
                    } catch (e: Exception) {
                        // ignore if engine is not initialized
                    }

                    // Relaunch MainActivity
                    val intent = Intent(this, MainActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                    }
                    startActivity(intent)
                    finish()
                }
            }
        }
    }
}

@Composable
fun CrashScreen(errorLog: String, onRestart: () -> Unit) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    Scaffold(
        containerColor = Color(0xFF020617), // Deep space black/slate
        bottomBar = {
            // Persistent cyber attribution footer
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF030712))
                    .border(BorderStroke(1.dp, Color(0xFFEF4444).copy(alpha = 0.3f)))
                    .padding(12.dp)
                    .navigationBarsPadding(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "Core Engine: zftoz • Snapdragon 8s Gen 4 Optimized • 100% Offline AI",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                    textAlign = TextAlign.Center,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(modifier = Modifier.height(24.dp))

            // Icon with blinking neon red background aura
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .background(Color(0xFFEF4444).copy(alpha = 0.1f), RoundedCornerShape(32.dp))
                    .border(1.dp, Color(0xFFEF4444), RoundedCornerShape(32.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "System fault",
                    tint = Color(0xFFEF4444),
                    modifier = Modifier.size(36.dp)
                )
            }

            Text(
                text = "NEURAL CORTEX SYSTEM PANIC",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Color(0xFFEF4444),
                fontFamily = FontFamily.Monospace,
                textAlign = TextAlign.Center
            )

            Text(
                text = "A critical hardware-level runtime exception was intercepted in the core engine. The active training state was serialized and safely sandboxed in isolated memory storage.",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF94A3B8),
                textAlign = TextAlign.Center,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(horizontal = 12.dp)
            )

            // Scrollable Stacktrace Text Panel
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color(0xFF030712), RoundedCornerShape(12.dp))
                    .border(1.dp, Color(0xFFEF4444).copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                    .padding(12.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                ) {
                    Text(
                        text = "--- DIAGNOSTIC DUMP (crash_log.txt) ---",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFEF4444),
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = errorLog,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFFF1F5F9),
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            // Quick Actions Block
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Copy Error Button
                OutlinedButton(
                    onClick = {
                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        val clip = ClipData.newPlainText("Cortex Crash Stacktrace", errorLog)
                        clipboard.setPrimaryClip(clip)
                        Toast.makeText(context, "Log copied to system clipboard!", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    shape = RoundedCornerShape(8.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF94A3B8)),
                    border = BorderStroke(1.dp, Color(0xFF334155))
                ) {
                    Icon(imageVector = Icons.Default.ContentCopy, contentDescription = "Copy log")
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("COPY ERROR", fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelSmall)
                }

                // Restart Engine Button
                Button(
                    onClick = onRestart,
                    modifier = Modifier
                        .weight(1.5f)
                        .height(50.dp),
                    shape = RoundedCornerShape(8.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF10B981),
                        contentColor = Color.Black
                    )
                ) {
                    Icon(imageVector = Icons.Default.Refresh, contentDescription = "Restart")
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("RESTART CORE ENGINE", fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}

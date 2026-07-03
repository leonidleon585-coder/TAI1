package com.example

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import com.example.ui.screens.TrainingHubScreen
import com.example.ui.screens.LocalChatScreen
import com.example.ui.screens.ValidationInsightsScreen
import com.example.ui.screens.SettingsScreen
import com.example.ui.theme.MyApplicationTheme
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.launch

/**
 * Available screen destinations in the side menu.
 */
enum class Screen(
    val title: String, 
    val route: String, 
    val icon: androidx.compose.ui.graphics.vector.ImageVector
) {
    TRAINING_HUB("Training Hub", "training_hub", Icons.Default.Terminal),
    LOCAL_CHAT("Local Chat", "local_chat", Icons.Default.Forum),
    VALIDATION_INSIGHTS("Validation & Insights", "validation_insights", Icons.Default.Analytics),
    SETTINGS("Settings", "settings", Icons.Default.Settings)
}

class MainActivity : ComponentActivity() {
    private lateinit var trainerEngine: TrainerEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        trainerEngine = TrainerEngine.getInstance(applicationContext)

        setContent {
            MyApplicationTheme {
                MainLayout(trainerEngine = trainerEngine)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainLayout(trainerEngine: TrainerEngine) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    val state by trainerEngine.state.collectAsState()

    // Setup navigation and side drawer
    val navController = rememberNavController()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = currentBackStackEntry?.destination?.route ?: Screen.TRAINING_HUB.route

    // File pickers launchers for custom base models and local text datasets
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

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = Color(0xFF0F172A),
                modifier = Modifier.width(300.dp)
            ) {
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = "TAI1 GEN CORTEX",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF10B981),
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp)
                )
                Text(
                    text = "zftoz Snapdragon Neural Core",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(horizontal = 24.dp).padding(bottom = 20.dp)
                )
                HorizontalDivider(color = Color(0xFF1E293B))
                Spacer(modifier = Modifier.height(12.dp))
                
                Screen.values().forEach { screen ->
                    NavigationDrawerItem(
                        icon = { 
                            Icon(
                                imageVector = screen.icon, 
                                contentDescription = screen.title, 
                                tint = if (currentRoute == screen.route) Color.Black else Color.LightGray
                            ) 
                        },
                        label = { 
                            Text(
                                text = screen.title, 
                                fontFamily = FontFamily.Monospace, 
                                fontWeight = FontWeight.SemiBold
                            ) 
                        },
                        selected = currentRoute == screen.route,
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        colors = NavigationDrawerItemDefaults.colors(
                            selectedContainerColor = Color(0xFF10B981),
                            selectedIconColor = Color.Black,
                            selectedTextColor = Color.Black,
                            unselectedContainerColor = Color.Transparent,
                            unselectedIconColor = Color.LightGray,
                            unselectedTextColor = Color.LightGray
                        ),
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { 
                        Text(
                            text = when(currentRoute) {
                                Screen.TRAINING_HUB.route -> "TRAINING ECOSYSTEM"
                                Screen.LOCAL_CHAT.route -> "NATIVE COGNITIVE CHAT"
                                Screen.VALIDATION_INSIGHTS.route -> "PARAMETER VALIDATION"
                                Screen.SETTINGS.route -> "SYSTEM CONFIG"
                                else -> "TAI1 ECOSYSTEM"
                            },
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        ) 
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(imageVector = Icons.Default.Menu, contentDescription = "Menu icon", tint = Color.White)
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF030712)
                    )
                )
            },
            bottomBar = {
                AttributionFooter()
            }
        ) { innerPadding ->
            NavHost(
                navController = navController,
                startDestination = Screen.TRAINING_HUB.route,
                modifier = Modifier.padding(innerPadding)
            ) {
                composable(Screen.TRAINING_HUB.route) {
                    TrainingHubScreen(
                        trainerEngine = trainerEngine,
                        state = state,
                        onPickModel = { modelPicker.launch("application/octet-stream") },
                        onPickDataset = { datasetPicker.launch("text/*") }
                    )
                }
                composable(Screen.LOCAL_CHAT.route) {
                    LocalChatScreen(
                        trainerEngine = trainerEngine,
                        state = state
                    )
                }
                composable(Screen.VALIDATION_INSIGHTS.route) {
                    ValidationInsightsScreen(
                        trainerEngine = trainerEngine,
                        state = state
                    )
                }
                composable(Screen.SETTINGS.route) {
                    SettingsScreen(
                        trainerEngine = trainerEngine,
                        state = state
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
            .background(Color(0xFF030712))
            .border(BorderStroke(1.dp, Color(0xFF1E293B)))
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

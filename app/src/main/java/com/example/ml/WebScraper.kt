package com.example.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

object WebScraper {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    /**
     * Fetches a webpage and returns sanitized prose text.
     */
    suspend fun fetchAndSanitize(urlStr: String): String = withContext(Dispatchers.IO) {
        val rawHtml = fetchRawHtml(urlStr)
        return@withContext sanitizeHtml(rawHtml)
    }

    /**
     * Fetches raw HTML of a webpage for link parsing and sanitization.
     */
    suspend fun fetchRawHtml(urlStr: String): String = withContext(Dispatchers.IO) {
        val url = if (!urlStr.startsWith("http://") && !urlStr.startsWith("https://")) {
            "https://$urlStr"
        } else {
            urlStr
        }

        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 (Linux; Android 10; Mobile) AIStudioTrainer/1.0")
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw Exception("HTTP Error: ${response.code} ${response.message}")
        }

        return@withContext response.body?.string() ?: throw Exception("Empty response body")
    }

    /**
     * Extracts up to 5 secondary links from the original domain paths in the HTML body.
     */
    fun extractLinks(html: String, baseUrl: String): List<String> {
        val linkRegex = "<a\\s+(?:[^>]*?\\s+)?href=\"([^\"]+)\"".toRegex(RegexOption.IGNORE_CASE)
        val matches = linkRegex.findAll(html)
        val links = mutableListOf<String>()
        
        val baseUri = try {
            java.net.URI(baseUrl)
        } catch (e: Exception) {
            null
        }
        val baseHost = baseUri?.host ?: ""

        for (match in matches) {
            var href = match.groupValues[1].trim()
            if (href.isEmpty() || href.startsWith("#") || href.startsWith("javascript:")) continue

            val resolvedUrl = try {
                if (href.startsWith("/")) {
                    val scheme = baseUri?.scheme ?: "https"
                    "$scheme://$baseHost$href"
                } else if (!href.startsWith("http://") && !href.startsWith("https://")) {
                    val scheme = baseUri?.scheme ?: "https"
                    val path = baseUri?.path ?: ""
                    val lastSlash = path.lastIndexOf('/')
                    val basePath = if (lastSlash >= 0) path.substring(0, lastSlash + 1) else "/"
                    "$scheme://$baseHost$basePath$href"
                } else {
                    href
                }
            } catch (e: Exception) {
                href
            }

            try {
                val resolvedUri = java.net.URI(resolvedUrl)
                val resolvedHost = resolvedUri.host ?: ""
                // Match domain and original path constraints
                if (resolvedHost.isNotEmpty() && (resolvedHost.endsWith(baseHost) || baseHost.endsWith(resolvedHost))) {
                    if (!links.contains(resolvedUrl)) {
                        links.add(resolvedUrl)
                    }
                }
            } catch (e: Exception) {
                // Ignore malformed links
            }
        }
        return links
    }

    /**
     * Strips HTML tags, script sections, styling blocks, and extracts pure text.
     */
    fun sanitizeHtml(html: String): String {
        var clean = html

        // Remove script tags and their contents
        val scriptRegex = "<script[^>]*?>[\\s\\S]*?</script>".toRegex(RegexOption.IGNORE_CASE)
        clean = scriptRegex.replace(clean, "")

        // Remove style tags and their contents
        val styleRegex = "<style[^>]*?>[\\s\\S]*?</style>".toRegex(RegexOption.IGNORE_CASE)
        clean = styleRegex.replace(clean, "")

        // Remove comment blocks
        val commentsRegex = "<!--[\\s\\S]*?-->".toRegex()
        clean = commentsRegex.replace(clean, "")

        // Strip general HTML tags
        val tagsRegex = "<[^>]*>".toRegex()
        clean = tagsRegex.replace(clean, " ")

        // Decode common HTML entities
        clean = clean
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
            .replace("&copy;", "")
            .replace("&reg;", "")

        // Collapse multiple whitespace/newlines into neat single spaces or simple carriage returns
        val multipleSpacesRegex = "\\s+".toRegex()
        clean = multipleSpacesRegex.replace(clean, " ")

        return clean.trim()
    }
}

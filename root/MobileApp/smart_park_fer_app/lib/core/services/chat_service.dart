import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:smart_park_fer_app/core/services/auth_service.dart';

class ChatService {
  static final ChatService _instance = ChatService._internal();
  factory ChatService() => _instance;
  ChatService._internal();

  static const String _baseUrl = 'http://100.85.74.87:8080/app';

  /// Analyzes a base64 encoded photo and returns the detected emotion.
  /// Requires authentication token.
  Future<String> analyzePhoto(String imageBase64) async {
    // 1. Retrieve the Access Token from AuthService
    String token;
    try {
      token = await AuthService().getAccessToken();
    } catch (e) {
      debugPrint('Authentication error in ChatService (analyzePhoto): $e');
      throw Exception('User is not authenticated or token retrieval failed');
    }

    final url = Uri.parse('$_baseUrl/analyzephoto');
    final headers = {
      'Authorization': 'Bearer $token',
      'Content-Type': 'application/json',
    };
    final body = jsonEncode({
      'image_base64': imageBase64,
    });

    debugPrint('--- OUTGOING API REQUEST (analyzephoto) ---');
    debugPrint('URL: $url');
    debugPrint('Headers: $headers');
    debugPrint('-----------------------------------------');

    try {
      final response = await http.post(
        url,
        headers: headers,
        body: body,
      ).timeout(const Duration(seconds: 60));

      debugPrint('Response status: ${response.statusCode}');
      debugPrint('Response body: ${response.body}');

      if (response.statusCode == 200) {
        final Map<String, dynamic> jsonData = jsonDecode(response.body);
        if (jsonData.containsKey('emotion')) {
          return jsonData['emotion'] as String;
        }
        throw Exception('Response does not contain "emotion" key');
      } else {
        debugPrint('Failed to analyze photo: ${response.statusCode}');
        throw Exception('Failed to analyze photo: ${response.statusCode}');
      }
    } on TimeoutException {
      debugPrint('Photo analysis timed out after 10 seconds');
      throw Exception('Photo analysis timed out. Please check your connection.');
    } catch (e) {
      debugPrint('Error analyzing photo: $e');
      rethrow;
    }
  }

  /// Sends a message/prompt to the agent and returns the agent's textual response.
  /// Requires authentication token.
  Future<String> askAgent(String message) async {
    // 1. Retrieve the Access Token from AuthService
    String token;
    try {
      token = await AuthService().getAccessToken();
    } catch (e) {
      debugPrint('Authentication error in ChatService (askAgent): $e');
      throw Exception('User is not authenticated or token retrieval failed');
    }

    final url = Uri.parse('$_baseUrl/askagent');
    final headers = {
      'Authorization': 'Bearer $token',
      'Content-Type': 'application/json',
    };
    final body = jsonEncode({
      'message': message,
    });

    debugPrint('--- OUTGOING API REQUEST (askagent) ---');
    debugPrint('URL: $url');
    debugPrint('Headers: $headers');
    debugPrint('--------------------------------------');

    try {
      final response = await http.post(
        url,
        headers: headers,
        body: body,
      ).timeout(const Duration(minutes: 10));

      debugPrint('Response status: ${response.statusCode}');
      debugPrint('Response body: ${response.body}');

      if (response.statusCode == 200) {
        final Map<String, dynamic> jsonData = jsonDecode(response.body);
        if (jsonData.containsKey('response')) {
          return jsonData['response'] as String;
        }
        throw Exception('Response does not contain "response" key');
      } else {
        debugPrint('Failed to ask agent: ${response.statusCode}');
        throw Exception('Failed to ask agent: ${response.statusCode}');
      }
    } on TimeoutException {
      debugPrint('Ask agent timed out after 15 seconds');
      throw Exception('Ask agent timed out. Please check your connection.');
    } catch (e) {
      debugPrint('Error asking agent: $e');
      rethrow;
    }
  }
}

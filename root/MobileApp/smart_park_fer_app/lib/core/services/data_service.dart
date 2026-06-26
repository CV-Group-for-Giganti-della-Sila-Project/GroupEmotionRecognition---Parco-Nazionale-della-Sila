import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:smart_park_fer_app/models/emotion_data.dart';
import 'package:smart_park_fer_app/core/services/auth_service.dart';

class DataService {
  static final DataService _instance = DataService._internal();
  factory DataService() => _instance;
  DataService._internal();

  static const String _baseUrl = 'http://100.85.74.87:8080/app/data';

  /// Fetches emotion data between two unix timestamps for a specific node.
  /// Requires authentication token.
  Future<List<EmotionData>> getEmotionData({
    required int start,
    required int end,
    required String nodename,
  }) async {
    // 1. Retrieve the Access Token from AuthService (handles refresh if necessary)
    String token;
    try {
      token = await AuthService().getAccessToken();
    } catch (e) {
      debugPrint('Authentication error in DataService: $e');
      throw Exception('User is not authenticated or token retrieval failed');
    }

    final url = Uri.parse('$_baseUrl/getbetweendates').replace(queryParameters: {
      'start': start.toString(),
      'end': end.toString(),
      'nodename': nodename,
    });

    final headers = {
      'Authorization': 'Bearer $token',
      'Content-Type': 'application/json',
    };

    debugPrint('--- OUTGOING API REQUEST ---');
    debugPrint('URL: $url');
    debugPrint('Headers: $headers');
    debugPrint('----------------------------');

    try {
      final response = await http.get(
        url,
        headers: headers,
      ).timeout(const Duration(seconds: 5));

      debugPrint('Response: ${response.body}');
      if (response.statusCode == 200) {
        final Map<String, dynamic> jsonData = jsonDecode(response.body);
        
        // The API returns a map where the key is the nodename
        if (jsonData.containsKey(nodename)) {
          final data = jsonData[nodename];
          
          if (data is List) {
            // Case where multiple records are returned for that node
            return data.map((item) => EmotionData.fromJson(item as Map<String, dynamic>)).toList();
          } else if (data is Map<String, dynamic>) {
            // Case where a single record is returned (as seen in your example)
            return [EmotionData.fromJson(data)];
          }
        }
        
        debugPrint('No data found for node: $nodename');
        return [];
      } else {
        debugPrint('Failed to fetch emotion data: ${response.statusCode}');
        debugPrint('Response: ${response.body}');
        throw Exception('Failed to load emotion data: ${response.statusCode}');
      }
    } on TimeoutException {
      debugPrint('Data retrieval timed out after 5 seconds');
      throw Exception('Data retrieval timed out. Please check your connection or server status.');
    } catch (e) {
      debugPrint('Error fetching emotion data: $e');
      rethrow;
    }
  }
}

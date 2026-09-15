import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class AuthService {
  static final AuthService _instance = AuthService._internal();
  factory AuthService() => _instance;
  AuthService._internal();

  static const String _endpoint = 'https://cognito-idp.eu-west-1.amazonaws.com/';
  // Passa il valore reale con: flutter build/run --dart-define=COGNITO_CLIENT_ID=xxxx
  static const String _clientId = String.fromEnvironment('COGNITO_CLIENT_ID');

  final _storage = const FlutterSecureStorage();

  String? _accessToken;
  String? _idToken;
  String? _refreshToken;
  Map<String, dynamic>? _userData;

  /// Initialize the service by loading persisted tokens
  Future<void> init() async {
    _accessToken = await _storage.read(key: 'accessToken');
    _idToken = await _storage.read(key: 'idToken');
    _refreshToken = await _storage.read(key: 'refreshToken');
    
    if (_accessToken != null) {
      // Validate the loaded token and fetch user data
      bool isValid = await validateToken();
      if (!isValid && _refreshToken != null) {
        await refreshSession();
      }
    }
  }

  Map<String, String> _getHeaders(String target) {
    return {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.$target',
    };
  }

  /// Registration (SignUp)
  Future<bool> signUp({
    required String username,
    required String password,
    required String email,
    required String birthdate,
    required String name,
  }) async {
    final body = jsonEncode({
      "ClientId": _clientId,
      "Username": username,
      "Password": password,
      "UserAttributes": [
        {"Name": "email", "Value": email},
        {"Name": "birthdate", "Value": birthdate},
        {"Name": "name", "Value": name}
      ]
    });

    try {
      final response = await http.post(
        Uri.parse(_endpoint),
        headers: _getHeaders('SignUp'),
        body: body,
      );

      if (response.statusCode == 200) {
        debugPrint('Sign up successful');
        return true;
      } else {
        debugPrint('Sign up failed: ${response.body}');
        return false;
      }
    } catch (e) {
      debugPrint('Sign up error: $e');
      return false;
    }
  }

  /// Confirmation code (ConfirmSignUp)
  Future<bool> confirmSignUp(String username, String code) async {
    final body = jsonEncode({
      "ClientId": _clientId,
      "Username": username,
      "ConfirmationCode": code
    });

    try {
      final response = await http.post(
        Uri.parse(_endpoint),
        headers: _getHeaders('ConfirmSignUp'),
        body: body,
      );

      if (response.statusCode == 200) {
        debugPrint('Confirmation successful');
        return true;
      } else {
        debugPrint('Confirmation failed: ${response.body}');
        return false;
      }
    } catch (e) {
      debugPrint('Confirmation error: $e');
      return false;
    }
  }

  /// Authentication token (InitiateAuth - USER_PASSWORD_AUTH)
  Future<bool> login(String username, String password) async {
    final body = jsonEncode({
      "AuthFlow": "USER_PASSWORD_AUTH",
      "ClientId": _clientId,
      "AuthParameters": {
        "USERNAME": username,
        "PASSWORD": password
      }
    });

    try {
      final response = await http.post(
        Uri.parse(_endpoint),
        headers: _getHeaders('InitiateAuth'),
        body: body,
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _accessToken = data['AuthenticationResult']['AccessToken'];
        _idToken = data['AuthenticationResult']['IdToken'];
        _refreshToken = data['AuthenticationResult']['RefreshToken'];
        
        // Persist tokens
        await _storage.write(key: 'accessToken', value: _accessToken);
        await _storage.write(key: 'idToken', value: _idToken);
        await _storage.write(key: 'refreshToken', value: _refreshToken);

        // Fetch user data immediately after login
        await validateToken();
        
        debugPrint('Login successful');
        return true;
      } else {
        debugPrint('Login failed: ${response.body}');
        return false;
      }
    } catch (e) {
      debugPrint('Login error: $e');
      return false;
    }
  }

  /// Refresh token (InitiateAuth - REFRESH_TOKEN_AUTH)
  Future<bool> refreshSession() async {
    if (_refreshToken == null) return false;

    final body = jsonEncode({
      "AuthFlow": "REFRESH_TOKEN_AUTH",
      "ClientId": _clientId,
      "AuthParameters": {
        "REFRESH_TOKEN": _refreshToken
      }
    });

    try {
      final response = await http.post(
        Uri.parse(_endpoint),
        headers: _getHeaders('InitiateAuth'),
        body: body,
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _accessToken = data['AuthenticationResult']['AccessToken'];
        _idToken = data['AuthenticationResult']['IdToken'];
        
        await _storage.write(key: 'accessToken', value: _accessToken);
        await _storage.write(key: 'idToken', value: _idToken);

        // Refresh token might not be returned, keep the old one
        if (data['AuthenticationResult']['RefreshToken'] != null) {
          _refreshToken = data['AuthenticationResult']['RefreshToken'];
          await _storage.write(key: 'refreshToken', value: _refreshToken);
        }
        
        debugPrint('Token refresh successful');
        return true;
      } else {
        debugPrint('Token refresh failed: ${response.body}');
        // If refresh fails, tokens might be totally invalid
        await logout();
        return false;
      }
    } catch (e) {
      debugPrint('Token refresh error: $e');
      return false;
    }
  }

  /// Logout globale (GlobalSignOut)
  Future<void> logout() async {
    if (_accessToken != null) {
      final body = jsonEncode({
        "AccessToken": _accessToken
      });

      try {
        await http.post(
          Uri.parse(_endpoint),
          headers: _getHeaders('GlobalSignOut'),
          body: body,
        );
      } catch (e) {
        debugPrint('GlobalSignOut error: $e');
      }
    }

    // Always clear local data and storage
    _accessToken = null;
    _idToken = null;
    _refreshToken = null;
    _userData = null;
    await _storage.deleteAll();
    debugPrint('Local session cleared');
  }

  /// Token validation & User retrieval (GetUser)
  Future<bool> validateToken() async {
    if (_accessToken == null) return false;

    final body = jsonEncode({
      "AccessToken": _accessToken
    });

    try {
      final response = await http.post(
        Uri.parse(_endpoint),
        headers: _getHeaders('GetUser'),
        body: body,
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        
        final Map<String, dynamic> profile = {};
        if (data['UserAttributes'] != null) {
          for (var attr in data['UserAttributes']) {
            profile[attr['Name']] = attr['Value'];
          }
        }
        profile['Username'] = data['Username'];
        
        _userData = profile;
        return true;
      }
      return false;
    } catch (e) {
      debugPrint('Token validation error: $e');
      return false;
    }
  }

  /// Returns the ID Token. Attempts to refresh if validation fails.
  Future<String> getToken() async {
    await _ensureValidSession();

    if (_idToken != null) {
      return _idToken!;
    }
    
    throw Exception('Token retrieval bad: No token found');
  }

  /// Returns the Access Token. Attempts to refresh if validation fails.
  Future<String> getAccessToken() async {
    await _ensureValidSession();

    if (_accessToken != null) {
      return _accessToken!;
    }
    
    throw Exception('Token retrieval bad: No token found');
  }

  /// Ensures a valid session exists by validating/refreshing tokens.
  Future<void> _ensureValidSession() async {
    bool isValid = await validateToken();

    if (!isValid) {
      bool refreshed = await refreshSession();
      if (!refreshed) {
        throw Exception('Token retrieval bad: Authentication expired');
      }
      await validateToken();
    }
  }

  String? get currentAccessToken => _accessToken;
  Map<String, dynamic>? get userData => _userData;
  bool get isAuthenticated => _accessToken != null;
}

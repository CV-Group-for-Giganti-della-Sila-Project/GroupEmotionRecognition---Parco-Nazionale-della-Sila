import 'package:flutter/material.dart';
import 'package:smart_park_fer_app/core/constants/colors.dart';
import 'package:smart_park_fer_app/core/services/auth_service.dart';
import 'package:smart_park_fer_app/screens/auth/login_screen.dart';
import 'package:smart_park_fer_app/screens/main_navigation_screen.dart';
import 'package:timezone/data/latest.dart' as tz;
import 'package:flutter/services.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  
  tz.initializeTimeZones();
  
  final authService = AuthService();
  await authService.init();
  
  runApp(ForestPulseApp(initialLoggedIn: authService.isAuthenticated));
}

class ForestPulseApp extends StatefulWidget {
  final bool initialLoggedIn;
  const ForestPulseApp({super.key, required this.initialLoggedIn});

  @override
  State<ForestPulseApp> createState() => _ForestPulseAppState();
}

class _ForestPulseAppState extends State<ForestPulseApp> {
  late bool _isLoggedIn;

  @override
  void initState() {
    super.initState();
    _isLoggedIn = widget.initialLoggedIn;
  }

  void _handleLogin() {
    setState(() {
      _isLoggedIn = true;
    });
  }

  void _handleLogout() {
    setState(() {
      _isLoggedIn = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
      ),
      child: MaterialApp(
        title: 'Forest Pulse',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          fontFamily: 'Roboto',
          colorScheme: ColorScheme.fromSeed(
            seedColor: AppColors.forestGreen,
          ),
          appBarTheme: const AppBarTheme(
            systemOverlayStyle: SystemUiOverlayStyle(
              statusBarColor: Colors.transparent,
              statusBarIconBrightness: Brightness.dark,
            ),
          ),
        ),
        home: _isLoggedIn 
            ? MainNavigationScreen(onLogout: _handleLogout) 
            : LoginScreen(onLoginSuccess: _handleLogin),
      ),
    );
  }
}

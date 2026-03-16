#ifndef AppName
  #define AppName "ICE CREAMS Studio"
#endif
#ifndef AppVersion
  #define AppVersion "1.0.1"
#endif
#ifndef AppDirName
  #define AppDirName "ICE_CREAMS_Studio"
#endif
#ifndef AppExeName
  #define AppExeName "ICE_CREAMS_Studio.exe"
#endif

[Setup]
AppId={{8D619190-15F4-4C35-8A31-B94BDBCB5D48}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ICE CREAMS
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=ICE_CREAMS_Installer_{#StringChange(AppVersion, ".", "_")}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\{#AppDirName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; Ensure the installed executable launches elevated (Run as administrator).
Root: HKLM; Subkey: "Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers"; ValueType: string; ValueName: "{app}\{#AppExeName}"; ValueData: "~ RUNASADMIN"; Flags: uninsdeletevalue

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

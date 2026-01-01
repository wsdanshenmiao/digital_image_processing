set_project("digital_image_processing")

set_languages("cxx23")
set_encodings("utf-8")

add_rules("mode.debug", "mode.release")

if is_mode("debug") then 
    binDir = path.join(os.projectdir(), "bin/debug/")
    set_symbols("debug")
    set_optimize("none")
else 
    binDir = path.join(os.projectdir(), "bin/release/")
end

rule("asset_copy")
    after_build(
        function(target)
            files = path.join(target:scriptdir(), "/asset");
            -- 判断文件是否存在
            if(os.exists(files)) then
                os.cp(files, target:targetdir())
            end
        end)
rule_end()

targetName = "digital_image_processing"
target(targetName)
    set_kind("binary")
    set_targetdir(path.join(binDir, targetName))

    add_files("test/*.cpp", "src/*.cpp")
    add_headerfiles("include/**.h", "thirdparty/**.h")
    
    add_includedirs("include", "thirdparty")
    
    add_rules("asset_copy")
target_end()

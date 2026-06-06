"""
Run this ONCE on your Mac to dump the exact page structure.
Then paste the output back so the scraper can be fixed properly.

    python3 inspect_page.py
"""
import asyncio
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)  # visible
        page = await browser.new_page()
        
        print("Loading page...")
        await page.goto("https://yonsei-mileage-helper.vercel.app", 
                       wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        print("\n=== PAGE TITLE ===")
        print(await page.title())

        print("\n=== ALL INPUTS ===")
        inputs = await page.query_selector_all("input, textarea")
        print(f"Found {len(inputs)} input(s)")
        for i, inp in enumerate(inputs):
            info = await page.evaluate("""el => ({
                tag:         el.tagName,
                type:        el.type,
                id:          el.id,
                name:        el.name,
                placeholder: el.placeholder,
                class:       el.className,
                value:       el.value,
                maxlength:   el.maxLength,
                visible:     el.offsetParent !== null,
                rect:        el.getBoundingClientRect().width + 'x' + el.getBoundingClientRect().height
            })""", inp)
            print(f"  [{i}] {info}")

        print("\n=== ALL BUTTONS ===")
        buttons = await page.query_selector_all("button, [role='button'], [onClick]")
        print(f"Found {len(buttons)} button(s)")
        for i, btn in enumerate(buttons):
            info = await page.evaluate("""el => ({
                tag:  el.tagName,
                text: el.innerText.trim().slice(0, 50),
                id:   el.id,
                class:el.className.slice(0, 60),
            })""", btn)
            print(f"  [{i}] {info}")

        print("\n=== FULL HTML (first 5000 chars) ===")
        html = await page.content()
        print(html[:5000])

        print("\n=== FULL HTML (5000-10000) ===")
        print(html[5000:10000])

        # Now try filling in manually
        print("\n\n=== TRYING TO FILL CAS3205 / 01 / 00 ===")
        inputs_fresh = await page.query_selector_all("input")
        if inputs_fresh:
            await inputs_fresh[0].click()
            await inputs_fresh[0].fill("CAS3205")
            val = await inputs_fresh[0].input_value()
            print(f"After fill('CAS3205'): value='{val}'")
            
            # Try typing instead
            await inputs_fresh[0].click(click_count=3)
            await page.keyboard.type("CAS3205")
            val2 = await inputs_fresh[0].input_value()
            print(f"After keyboard.type('CAS3205'): value='{val2}'")

        input("\nPress Enter to close browser...")
        await browser.close()

asyncio.run(inspect())
